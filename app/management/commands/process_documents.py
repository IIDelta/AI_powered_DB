import os
import json
import requests
from docx import Document
import PyPDF2

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

# Import the models from your app
from app.models import (
    Project, Sponsor, Deliverable, TherapeuticArea, IngredientCategory,
    Ingredient, ResponsibleParty, RouteOfAdmin, Demographics
)

# --- Helper Functions for Text Extraction ---

def extract_text_from_docx(file_path):
    """Extracts raw text from a .docx file."""
    try:
        doc = Document(file_path)
        return "\\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Error reading docx {file_path}: {e}")
        return None

def extract_text_from_pdf(file_path):
    """Extracts raw text from a .pdf file."""
    try:
        text = ""
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\\n"
        return text
    except Exception as e:
        print(f"Error reading pdf {file_path}: {e}")
        return None

# --- LLM Interaction ---

def call_phi3_for_extraction(document_text):
    """
    Sends document text to a local Ollama server running Phi-3
    and asks for structured JSON output.
    """
    # The prompt instructs the LLM to act as an extractor and return JSON.
    # Note: We are using snake_case for the keys to match Django model fields.
    prompt = f"""
    You are an expert data extraction assistant. Analyze the following document text and extract the required information.
    Your response MUST be a single, valid JSON object and nothing else. Do not include any explanatory text before or after the JSON.

    The JSON object must have the following keys: "project_id", "project_name", "sponsor_name", "deliverables",
    "project_status", "therapeutic_areas", "ingredient_categories", "ingredients", "responsible_party",
    "route_of_admin", "demographics".

    - For fields that can contain multiple values (like 'ingredients' or 'therapeutic_areas'), return a JSON array of strings.
    - If a value for a specific field cannot be found, use a JSON null value for that key.

    Document Text:
    ---
    {document_text[:8000]}
    ---

    JSON Output:
    """

    ollama_api_url = "http://localhost:11434/api/generate"
    payload = {
        "model": "phi3",
        "prompt": prompt,
        "format": "json",
        "stream": False
    }

    try:
        response = requests.post(ollama_api_url, json=payload, timeout=180) # 3-minute timeout
        response.raise_for_status()
        response_data = response.json()
        json_string = response_data.get("response", "{}")
        return json.loads(json_string)

    except requests.exceptions.RequestException as e:
        print(f"API Error: Could not connect to Ollama server. Is it running? Error: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: The model did not return valid JSON. Response: {json_string}. Error: {e}")
        return None

# --- Django Management Command ---

class Command(BaseCommand):
    help = 'Scans a directory for documents, extracts information using a local LLM, and populates the database.'

    def add_arguments(self, parser):
        parser.add_argument('directory_path', type=str, help='The path to the directory containing documents.')
        parser.add_argument(
            '--update',
            action='store_true',
            help='Update existing projects if a matching project_id is found.',
        )

    @transaction.atomic
    def handle(self, *args, **options):
        directory_path = options['directory_path']
        update_existing = options['update']

        if not os.path.isdir(directory_path):
            raise CommandError(f"Directory not found: '{directory_path}'")

        self.stdout.write(self.style.SUCCESS(f"Starting to process documents in '{directory_path}'..."))

        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            raw_text = None

            if filename.lower().endswith('.docx'):
                self.stdout.write(f"Processing DOCX: {filename}")
                raw_text = extract_text_from_docx(file_path)
            elif filename.lower().endswith('.pdf'):
                self.stdout.write(f"Processing PDF: {filename}")
                raw_text = extract_text_from_pdf(file_path)

            if not raw_text:
                continue

            self.stdout.write("  -> Extracting data with local LLM (this may take a moment)...")
            extracted_data = call_phi3_for_extraction(raw_text)

            if not extracted_data:
                self.stderr.write(self.style.ERROR(f"  -> Failed to extract data from {filename}."))
                continue

            try:
                # Check if project already exists
                project_id_val = extracted_data.get('project_id')
                if not project_id_val:
                    self.stderr.write(self.style.WARNING(f"  -> Skipping {filename}: no 'project_id' found in extracted data."))
                    continue

                project, created = Project.objects.get_or_create(
                    project_id=project_id_val,
                    defaults={'project_name': extracted_data.get('project_name', '')}
                )

                if created:
                    self.stdout.write(self.style.SUCCESS(f"  -> Created new project: {project.project_id}"))
                elif update_existing:
                    self.stdout.write(self.style.NOTICE(f"  -> Found existing project: {project.project_id}. Updating..."))
                else:
                    self.stdout.write(self.style.WARNING(f"  -> Skipping existing project: {project.project_id}. Use --update to overwrite."))
                    continue
                
                # Update simple fields
                project.project_name = extracted_data.get('project_name', project.project_name)
                project.project_status = extracted_data.get('project_status', project.project_status)

                # --- Handle relationships (get or create related objects) ---

                # Sponsor
                if sponsor_name := extracted_data.get('sponsor_name'):
                    sponsor, _ = Sponsor.objects.get_or_create(name=sponsor_name)
                    project.sponsor = sponsor
                
                # Responsible Party
                if party_name := extracted_data.get('responsible_party'):
                    party, _ = ResponsibleParty.objects.get_or_create(name=party_name)
                    project.responsible_party = party
                    
                # Route of Admin
                if route_name := extracted_data.get('route_of_admin'):
                    route, _ = RouteOfAdmin.objects.get_or_create(name=route_name)
                    project.route_of_admin = route

                project.save() # Save foreign key updates

                # --- Handle Many-to-Many relationships ---
                
                # Deliverables
                if deliverables_list := extracted_data.get('deliverables', []):
                    project.deliverables.clear()
                    for item_name in deliverables_list:
                        item, _ = Deliverable.objects.get_or_create(name=item_name)
                        project.deliverables.add(item)
                
                # Therapeutic Areas
                if ta_list := extracted_data.get('therapeutic_areas', []):
                    project.therapeutic_areas.clear()
                    for item_name in ta_list:
                        item, _ = TherapeuticArea.objects.get_or_create(name=item_name)
                        project.therapeutic_areas.add(item)
                
                # Ingredient Categories
                if cat_list := extracted_data.get('ingredient_categories', []):
                    project.ingredient_categories.clear()
                    for item_name in cat_list:
                        item, _ = IngredientCategory.objects.get_or_create(name=item_name)
                        project.ingredient_categories.add(item)
                        
                # Ingredients
                if ing_list := extracted_data.get('ingredients', []):
                    project.ingredients.clear()
                    for item_name in ing_list:
                        item, _ = Ingredient.objects.get_or_create(name=item_name)
                        project.ingredients.add(item)
                        
                # Demographics
                if demo_list := extracted_data.get('demographics', []):
                    project.demographics.clear()
                    for item_name in demo_list:
                        item, _ = Demographics.objects.get_or_create(name=item_name)
                        project.demographics.add(item)

                self.stdout.write(self.style.SUCCESS(f"  -> Successfully processed and saved data for {project.project_id}"))

            except Exception as e:
                self.stderr.write(self.style.ERROR(f"  -> Database Error for {filename}: {e}"))

        self.stdout.write(self.style.SUCCESS("\nProcessing complete."))
