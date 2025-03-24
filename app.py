import time
from flask import Flask, render_template, request, jsonify, session, Response
import os
from pathlib import Path
import threading
import time
import requests
import uuid
from typing import Dict, Any
from collections import defaultdict
from dotenv import load_dotenv
from functools import wraps
from utils.logger import configure_logger

# Load environment variables
load_dotenv()

# Initialize logger
logger = configure_logger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', os.urandom(32))  # Use random key if not set
app.config['SESSION_COOKIE_SECURE'] = True  # Only send cookies over HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True  # Prevent JavaScript access to session cookie
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # Protect against CSRF

# Get the absolute path to the output directory
BASE_DIR = Path(__file__).resolve().parent
app.config['UPLOAD_FOLDER'] = str(BASE_DIR / 'output')
app.config['METADATA_FILE'] = str(BASE_DIR / 'output' / 'metadata.json')

# Security constants
REQUEST_TIMEOUT = 60  # Timeout for requests to the AI service

AI_SERVICE_AUTH_TOKEN = os.environ.get('AI_SERVICE_AUTH_TOKEN')
if not AI_SERVICE_AUTH_TOKEN:
    raise ValueError("AI_SERVICE_AUTH_TOKEN environment variable is required")

# Configuration constants
AI_SERVICE_URL = "http://localhost:5001"  # URL of the AI service

# Rate limiting settings
RATE_LIMIT_WINDOW = 60  # Window size in seconds
MAX_REQUESTS_PER_USER = 3  # Max requests per user per window
MAX_CONCURRENT_JOBS = 2  # Max concurrent jobs per user
GLOBAL_RATE_LIMIT = 10  # Max total requests per minute across all users

# Rate limiting storage
user_request_times = defaultdict(list)  # Store request timestamps per user
user_concurrent_jobs = defaultdict(int)  # Track concurrent jobs per user
last_cleanup = time.time()  # Last time we cleaned up old request records
request_lock = threading.Lock()  # Lock for thread-safe rate limiting

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

def validate_request_data(data: Dict[str, Any]) -> tuple[bool, str]:
    """Validate request data"""
    try:
        # Validate prompt
        prompt = data.get('prompt', '').strip()
        if not prompt:
            return False, "Prompt is required"
        if len(prompt) > 200:  # Max prompt length
            return False, "Prompt too long"

        # Validate dimensions
        height = int(data.get('height', 512))
        width = int(data.get('width', 512))
        if not (64 <= height <= 1024 and 64 <= width <= 1024):
            return False, "Invalid dimensions"

        # Validate steps
        steps = int(data.get('num_inference_steps', 50))
        if not (1 <= steps <= 100):
            return False, "Invalid number of steps"

        return True, ""
    except (ValueError, TypeError) as e:
        return False, f"Invalid input format: {str(e)}"

def cleanup_old_requests():
    """Remove request records older than the rate limit window"""
    global last_cleanup
    current_time = time.time()
    
    # Only clean up every 10 seconds at most
    if current_time - last_cleanup < 10:
        return
        
    with request_lock:
        cutoff_time = current_time - RATE_LIMIT_WINDOW
        for user_id in list(user_request_times.keys()):
            # Keep only requests within the window
            user_request_times[user_id] = [
                t for t in user_request_times[user_id] 
                if t > cutoff_time
            ]
            # Remove user if they have no recent requests
            if not user_request_times[user_id]:
                del user_request_times[user_id]
        last_cleanup = current_time

def check_rate_limit():
    """Check if the request should be rate limited"""
    cleanup_old_requests()
    
    user_id = session.get('user_id')
    if not user_id:
        return False, "No user session"
        
    current_time = time.time()
    
    with request_lock:
        # Check user's request count
        user_requests = user_request_times[user_id]
        if len(user_requests) >= MAX_REQUESTS_PER_USER:
            return False, f"Rate limit exceeded. Please wait {RATE_LIMIT_WINDOW} seconds between requests."
            
        # Check user's concurrent jobs
        if user_concurrent_jobs[user_id] >= MAX_CONCURRENT_JOBS:
            return False, "Too many concurrent jobs. Please wait for your current generations to complete."
            
        # Check global rate limit
        total_requests = sum(len(times) for times in user_request_times.values())
        if total_requests >= GLOBAL_RATE_LIMIT:
            return False, "Server is busy. Please try again later."
            
        # Add the new request
        user_request_times[user_id].append(current_time)
        user_concurrent_jobs[user_id] += 1
        
    return True, ""

def rate_limit_complete(user_id):
    """Call this when a job completes to decrement the concurrent job counter"""
    with request_lock:
        user_concurrent_jobs[user_id] = max(0, user_concurrent_jobs[user_id] - 1)

def check_ai_service():
    """Check if the AI service is running"""
    try:
        response = requests.get(
            f"{AI_SERVICE_URL}/health",
            headers={'X-Auth-Token': AI_SERVICE_AUTH_TOKEN}
        )
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def initialize_service():
    """Initialize the Web Interface components"""
    print("\n" + "="*50)
    print("🚀 InfernalForge Web Interface")
    print("="*50 + "\n")
    start_time = time.time()

    print("📦 Initializing...")
    print(f"✓ Dependencies loaded ({time.time() - start_time:.2f}s)")
    
    print("\n⚙️  Configuration")
    print(f"✓ Output directory: {app.config['UPLOAD_FOLDER']}")
    
    print("\n🔍 Service Check")
    if check_ai_service():
        print("✓ AI service connected")
    else:
        print("❌ AI service not available")
        print("  • Please start: python ai_service.py")
    
    print("\n🌐 Service Status")
    print(f"✓ Running on http://localhost:5000")
    print(f"✓ Total startup: {time.time() - start_time:.2f}s")
    print("\n" + "="*50 + "\n")

@app.route('/')
def index():
    """Serve the main page"""
    # Generate a unique user ID if not exists
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    return render_template('index.html', 
                         ai_service_auth_token=AI_SERVICE_AUTH_TOKEN,
                         max_requests=MAX_REQUESTS_PER_USER,
                         rate_limit_window=RATE_LIMIT_WINDOW)

@app.route('/generate', methods=['POST'])
def generate_image():
    try:
        # Validate session
        if 'user_id' not in session:
            return jsonify({
                'success': False,
                'error': "No user session found"
            }), 401

        # Check rate limit
        can_proceed, rate_limit_error = check_rate_limit()
        if not can_proceed:
            return jsonify({
                'success': False,
                'error': rate_limit_error
            }), 429

        # Validate request data
        data = request.json
        if not data:
            return jsonify({
                'success': False,
                'error': "No data provided"
            }), 400

        try:
            # Rest of the existing generate_image code...
            is_valid, error_message = validate_request_data(data)
            if not is_valid:
                return jsonify({
                    'success': False,
                    'error': error_message
                }), 400

            # Add user_id to the request
            data['user_id'] = session['user_id']
            
            # Forward the request to the AI service with timeout and auth token
            response = requests.post(
                f"{AI_SERVICE_URL}/generate",
                json=data,
                timeout=REQUEST_TIMEOUT,
                headers={'X-Auth-Token': AI_SERVICE_AUTH_TOKEN}
            )
            response.raise_for_status()
            
            result = response.json()
            if result.get('success'):
                # Return the complete response from the AI service
                return jsonify(result)
            else:
                return jsonify({
                    'success': False,
                    'error': result.get('error', 'Unknown error')
                }), 500
        finally:
            # Always decrement the concurrent job counter
            rate_limit_complete(session['user_id'])

    except requests.exceptions.Timeout:
        return jsonify({
            'success': False,
            'error': "Request timed out"
        }), 504
    except requests.exceptions.RequestException as e:
        return jsonify({
            'success': False,
            'error': f"Error communicating with AI service: {str(e)}"
        }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/output/<uuid:file_id>')
@require_auth_token
def serve_image(file_id):
    """Proxy image requests to the AI service"""
    logger.info(f"Received image request for file_id: {file_id}")
    try:
        # Forward the request to the AI service
        url = f"{AI_SERVICE_URL}/output/{file_id}"
        logger.info(f"Forwarding request to AI service: {url}")
        response = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
            stream=True,  # Stream the response
            headers={'X-Auth-Token': AI_SERVICE_AUTH_TOKEN}
        )
        response.raise_for_status()
        
        # Get the content type from the response
        content_type = response.headers.get('content-type', 'image/png')
        logger.info(f"Received response with content-type: {content_type}")
        
        # Create a Flask response with the streamed content
        return Response(
            response.iter_content(chunk_size=8192),
            content_type=content_type,
            headers={
                'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0',
                'Pragma': 'no-cache',
                'Expires': '0'
            }
        )
    except requests.exceptions.RequestException as e:
        logger.error(f"Error proxying image request: {e}")
        logger.error(f"Request URL: {url}")
        return jsonify({'error': 'Image not found'}), 404

@app.route('/status/<job_id>')
@require_auth_token
def get_job_status(job_id):
    """Proxy status requests to the AI service"""
    try:
        response = requests.get(
            f"{AI_SERVICE_URL}/status/{job_id}",
            timeout=REQUEST_TIMEOUT,
            headers={'X-Auth-Token': AI_SERVICE_AUTH_TOKEN}
        )
        response.raise_for_status()
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        logger.error(f"Error checking job status: {e}")
        return jsonify({'error': 'Failed to check job status'}), 500

@app.after_request
def add_security_headers(response):
    """Add security headers to all responses"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.tailwindcss.com; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; img-src 'self' data: blob:;"
    return response

@app.route('/health')
def health_check():
    """Check if the AI service is healthy and responding"""
    logger.debug("Health check requested")
    try:
        # Try to connect to the AI service using the configured URL
        response = requests.get(
            f"{AI_SERVICE_URL}/health",
            timeout=2,
            headers={'X-Auth-Token': AI_SERVICE_AUTH_TOKEN}
        )
        logger.debug(f"AI service health check response: {response.status_code}, content: {response.text}")
        
        try:
            data = response.json()
        except ValueError:
            logger.error("Failed to parse AI service response as JSON")
            return jsonify({
                'status': 'unhealthy',
                'reason': 'Invalid response from AI service'
            }), 503

        # If we got a response and it's healthy, return healthy
        if response.status_code == 200 and data.get('status') == 'healthy':
            return jsonify({'status': 'healthy'})
        
        # If we got a response but it's unhealthy, include the reason
        reason = data.get('reason', 'AI service reported unhealthy')
        logger.warning(f"AI service reported unhealthy: {reason}")
        return jsonify({
            'status': 'unhealthy',
            'reason': reason
        }), 503

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to connect to AI service: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'reason': 'Failed to connect to AI service'
        }), 503
    except Exception as e:
        logger.error(f"Unexpected error during health check: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'reason': 'Internal server error'
        }), 503

if __name__ == '__main__':
    # Only print startup info in the main process
    if not os.environ.get('WERKZEUG_RUN_MAIN'):
        initialize_service()
    
    app.run(host='0.0.0.0', port=5000, debug=True) 