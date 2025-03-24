import time
from flask import Flask, request, jsonify, send_from_directory, render_template
import os
from pathlib import Path
from utils.pipeline import initialize_pipeline
from utils.image_utils import process_image
import re
from typing import Dict, Any
import uuid
from threading import Lock
from utils.logger import configure_logger
from utils.config import output_dir
from flask_cors import CORS
from functools import wraps
from dotenv import load_dotenv
import threading

# Load environment variables from .env file
load_dotenv()

logger = configure_logger(__name__)

# Initialize Flask app for the AI service
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:5000"]}})  # Enable CORS for specific origin

# Security settings
AI_SERVICE_AUTH_TOKEN = os.environ.get('AI_SERVICE_AUTH_TOKEN')
if not AI_SERVICE_AUTH_TOKEN:
    raise ValueError("AI_SERVICE_AUTH_TOKEN environment variable is required")

# Configure Flask for long-running requests
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching
app.config['PERMANENT_SESSION_LIFETIME'] = 300  # 5 minutes session lifetime
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max content length

# Get the absolute path to the output directory
BASE_DIR = Path(__file__).resolve().parent
app.config['UPLOAD_FOLDER'] = str(BASE_DIR / 'output')

# Ensure output directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global pipeline instance
pipeline = None
pipeline_lock = Lock()

# File mapping for UUID to actual paths
file_mapping = {}
file_mapping_lock = Lock()

# Job tracking
jobs = {}
jobs_lock = Lock()

# Security constants
MAX_PROMPT_LENGTH = 200  # Maximum length for prompts
MAX_DIMENSION = 1024    # Maximum image dimension
MIN_DIMENSION = 64      # Minimum image dimension
MAX_STEPS = 100         # Maximum number of inference steps
MIN_STEPS = 1           # Minimum number of inference steps
ALLOWED_CHARS = re.compile(r'^[a-zA-Z0-9\s\-_.,!?()]+$')  # Allowed characters in prompts

def require_auth_token(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check for token in headers or query parameters
        auth_header = request.headers.get('X-Auth-Token')
        auth_query = request.args.get('auth')
        auth_token = auth_header or auth_query
        
        logger.debug(f"Auth token from header: {auth_header}")
        logger.debug(f"Auth token from query: {auth_query}")
        
        if not auth_token or auth_token != AI_SERVICE_AUTH_TOKEN:
            logger.error("Authentication failed - token mismatch or missing")
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated_function

def initialize_service():
    """Initialize the AI service components"""
    print("\n🔄 Initializing AI Service...")
    
    # Create output directory
    print("  • Creating output directory...")
    os.makedirs(output_dir, exist_ok=True)
    print("    ✓ Output directory ready")
    
    # Initialize pipeline
    print("\n🧠 Initializing AI Pipeline")
    print("  • This may take up to 30 seconds...")
    start_time = time.time()
    
    try:
        global pipeline
        pipeline = initialize_pipeline()
        init_time = time.time() - start_time
        print(f"    ✓ Pipeline initialized in {init_time:.1f} seconds")
        return pipeline
    except Exception as e:
        print(f"    ❌ Pipeline initialization failed: {str(e)}")
        raise

def ensure_user_directory(user_id):
    """Ensure a user's directory exists and return its path"""
    user_dir = os.path.join(app.config['UPLOAD_FOLDER'], user_id)
    os.makedirs(user_dir, exist_ok=True)
    return user_dir

def validate_input(data: Dict[str, Any]) -> tuple[bool, str]:
    """
    Validate and sanitize input data.
    
    Args:
        data: Dictionary containing request data
        
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        # Validate prompt
        prompt = data.get('prompt', '').strip()
        if not prompt:
            return False, "Prompt is required"
        if len(prompt) > MAX_PROMPT_LENGTH:
            return False, f"Prompt too long (max {MAX_PROMPT_LENGTH} characters)"
        if not ALLOWED_CHARS.match(prompt):
            return False, "Prompt contains invalid characters"

        # Validate dimensions
        height = int(data.get('height', 512))
        width = int(data.get('width', 512))
        if not (MIN_DIMENSION <= height <= MAX_DIMENSION and MIN_DIMENSION <= width <= MAX_DIMENSION):
            return False, f"Dimensions must be between {MIN_DIMENSION} and {MAX_DIMENSION}"

        # Validate steps
        steps = int(data.get('num_inference_steps', 50))
        if not (MIN_STEPS <= steps <= MAX_STEPS):
            return False, f"Steps must be between {MIN_STEPS} and {MAX_STEPS}"

        # Validate user_id
        user_id = data.get('user_id')
        if not user_id:
            return False, "User ID is required"
        if not re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', user_id):
            return False, "Invalid user ID format"

        # Validate seed if provided
        seed = data.get('seed')
        if seed is not None:
            try:
                seed = int(seed)
                if not (0 <= seed <= 2**32 - 1):
                    return False, "Invalid seed value"
            except ValueError:
                return False, "Seed must be a valid integer"

        return True, ""

    except (ValueError, TypeError) as e:
        return False, f"Invalid input format: {str(e)}"

def get_pipeline():
    """Get the pipeline instance"""
    global pipeline
    if pipeline is None:
        logger.warning("Pipeline was not initialized during service startup")
        with pipeline_lock:
            if pipeline is None:  # Double-check after acquiring lock
                pipeline = initialize_pipeline()
    return pipeline

def process_job(job_id, data):
    """Process a job asynchronously"""
    try:
        # Get the pipeline
        pipe = get_pipeline()
        
        # Process the image
        image_path, seed = process_image(
            pipe=pipe,
            prompt=data['prompt'],
            height=data.get('height', 512),
            width=data.get('width', 512),
            num_inference_steps=data.get('num_inference_steps', 50),
            seed=data.get('seed'),
            output_dir=ensure_user_directory(data['user_id'])
        )

        # Get the relative path for the response
        relative_path = os.path.relpath(image_path, app.config['UPLOAD_FOLDER'])
        
        # Store the mapping
        with file_mapping_lock:
            file_mapping[job_id] = relative_path
            logger.info(f"Stored mapping: {job_id} -> {relative_path}")

        # Update job status
        with jobs_lock:
            jobs[job_id]['status'] = 'completed'
            jobs[job_id]['result'] = {
                'job_id': job_id,
                'image_path': f'/output/{job_id}',
                'seed': seed
            }
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {e}")
        with jobs_lock:
            jobs[job_id]['status'] = 'failed'
            jobs[job_id]['error'] = str(e)

@app.route('/output/<uuid:file_id>')
@require_auth_token
def serve_image(file_id):
    """Serve generated images from the output directory using UUID mapping"""
    logger.info(f"AI service received image request for file_id: {file_id}")
    try:
        with file_mapping_lock:
            if str(file_id) not in file_mapping:
                logger.error(f"File ID {file_id} not found in mapping")
                logger.error(f"Current mappings: {file_mapping}")
                return jsonify({'error': 'Image not found'}), 404
            actual_path = file_mapping[str(file_id)]
            logger.info(f"Serving file {actual_path} for UUID {file_id}")
            
        # Ensure the file exists
        full_path = os.path.join(app.config['UPLOAD_FOLDER'], actual_path)
        logger.info(f"Full file path: {full_path}")
        if not os.path.exists(full_path):
            logger.error(f"File not found at path: {full_path}")
            return jsonify({'error': 'Image not found'}), 404
            
        # Set cache control headers and serve the file
        response = send_from_directory(
            app.config['UPLOAD_FOLDER'],
            actual_path,
            mimetype='image/png'
        )
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        logger.info(f"Successfully serving file: {actual_path}")
        return response
    except Exception as e:
        logger.error(f"Error serving image {file_id}: {e}")
        return jsonify({'error': 'Image not found'}), 404

@app.route('/generate', methods=['POST'])
@require_auth_token
def generate():
    try:
        data = request.get_json()
        if not data or 'prompt' not in data or 'user_id' not in data:
            return jsonify({'error': 'Missing required fields'}), 400

        # Validate input
        is_valid, error_message = validate_input(data)
        if not is_valid:
            return jsonify({'error': error_message}), 400

        # Generate a job ID
        job_id = str(uuid.uuid4())
        
        # Initialize job status
        with jobs_lock:
            jobs[job_id] = {
                'status': 'processing',
                'started_at': time.time(),
                'data': data
            }

        # Start processing in background
        thread = threading.Thread(target=process_job, args=(job_id, data))
        thread.daemon = True
        thread.start()

        # Return job ID immediately, no need to wait for processing to complete
        return jsonify({
            'success': True,
            'job_id': job_id,
            'status': 'processing'
        })

    except Exception as e:
        logger.error(f"Unexpected error in /generate endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/status/<job_id>')
@require_auth_token
def get_job_status(job_id):
    """Get the status of a job"""
    try:
        with jobs_lock:
            if job_id not in jobs:
                return jsonify({'error': 'Job not found'}), 404
            
            job = jobs[job_id]
            response = {
                'status': job['status'],
                'started_at': job['started_at']
            }
            
            if job['status'] == 'completed':
                response.update(job['result'])
            elif job['status'] == 'failed':
                response['error'] = job['error']
                
            return jsonify(response)
    except Exception as e:
        logger.error(f"Error getting job status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
@require_auth_token
def health_check():
    """Health check endpoint"""
    try:
        # Try to get the pipeline to verify it's working
        pipe = get_pipeline()
        return jsonify({'status': 'healthy'})
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@app.route('/')
def index():
    """Serve the informational page for the AI service"""
    try:
        # Check health status
        pipe = get_pipeline()
        status = 'healthy'
        status_class = 'operational'
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        status = 'unhealthy'
        status_class = 'error'
    
    return render_template('ai_service.html', status=status, status_class=status_class)

if __name__ == '__main__':
    try:
        # Initialize the service with all configurations
        initialize_service()
        
        # Start the Flask app
        app.run(
            host='0.0.0.0',
            port=5001,
            debug=True,
            use_reloader=False  # Disable reloader to prevent duplicate pipeline initialization
        )
    except Exception as e:
        logger.error(f"Failed to start AI service: {e}")
        raise 