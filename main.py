import os
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed
from google import genai
import replicate
from authlib.integrations.starlette_client import OAuth
from starlette.middleware.sessions import SessionMiddleware
from datetime import datetime, timedelta
from jose import jwt, JWTError

load_dotenv()

# Environment Variables
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
REPLICATE_TOKEN = os.getenv("REPLICATE_API_TOKEN")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
SECRET_KEY = os.getenv("SECRET_KEY", "default_secret_fida_key")
ALGORITHM = "HS256"

app = FastAPI(title="Herry AI Secure Backend Services")

# Session Middleware (OAuth handling ke liye required hai)
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

# OAuth Setup
oauth = OAuth()
oauth.register(
    name='google',
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

# Gemini Client
gemini_client = None
if GEMINI_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_KEY)
    except Exception as e:
        print(f"Gemini Init Error: {e}")

# Models
class ChatRequest(BaseModel):
    prompt: str

class ImageRequest(BaseModel):
    prompt: str

# Helper Functions
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=7)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized: Login required")
    token = auth_header.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2), reraise=True)
def safe_replicate_generate(prompt: str):
    return replicate.run("black-forest-labs/flux-schnell", input={"prompt": prompt})

# ==================== AUTH ROUTES ====================

@app.get("/login/google")
async def login_google(request: Request):
    """User ko Google Sign-in page par redirect karta hai"""
    redirect_uri = request.url_for('auth_callback')
    return await oauth.google.authorize_redirect(request, redirect_uri)

@app.get("/auth/callback")
async def auth_callback(request: Request):
    """Google Login ke baad JWT Token generate karta hai"""
    try:
        token = await oauth.google.authorize_access_token(request)
        user_info = token.get('userinfo')
        if not user_info:
            raise HTTPException(status_code=400, detail="Failed to fetch user profile")
        
        # User details se JWT Access Token banayein
        jwt_token = create_access_token({
            "email": user_info["email"],
            "name": user_info.get("name", ""),
            "picture": user_info.get("picture", "")
        })
        
        return {
            "status": "success",
            "message": "Login Successful!",
            "access_token": jwt_token,
            "user": user_info
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Authentication Error: {str(e)}")

# ==================== SECURE PROTECTED ROUTES ====================

@app.post("/api/chat")
def chat_with_herry(req: ChatRequest, current_user: dict = Depends(get_current_user)):
    """Only Logged-in users can access this"""
    if not gemini_client:
        raise HTTPException(status_code=500, detail="Gemini API Key is missing.")
    
    try:
        user_email = current_user.get("email")
        prompt_with_context = f"User Email ({user_email}): {req.prompt}"
        
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt_with_context,
        )
        return {"user": user_email, "response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini Error: {str(e)}")

@app.post("/api/generate-image")
def generate_image(req: ImageRequest, current_user: dict = Depends(get_current_user)):
    """Only Logged-in users can generate images"""
    if not REPLICATE_TOKEN:
        raise HTTPException(status_code=500, detail="Replicate Token is missing.")
    
    try:
        image_urls = safe_replicate_generate(req.prompt)
        return {"status": "success", "user": current_user.get("email"), "images": image_urls}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to generate image after retries")