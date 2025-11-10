import os
import traceback
from datetime import datetime
from datetime import timezone
from dotenv import load_dotenv
import logging
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from intent_classifier import IntentClassifier
from db.engine import get_mongo_collection
from app.auth import verify_token

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Read environment mode (defaults to prod for safety)
ENV = os.getenv("ENV", "prod").lower()
logger.info(f"Running in {ENV} mode")

# Initialize FastAPI app
app = FastAPI(
    title="Basic ML App",
    description="A basic ML app with intent classification",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:3000",
        "https://meusite.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database connection (with fallback)
collection = None
try:
    collection = get_mongo_collection(f"{ENV.upper()}_intent_logs")
    logger.info(f"Database connection established: {ENV.upper()}_intent_logs")
except Exception as e:
    logger.error(f"Failed to connect to database: {str(e)}")
    logger.warning("⚠️  Running WITHOUT database persistence")
    collection = None  # Explicitly set to None


# Conditional authentication dependency
def conditional_auth(request: Request):
    """Returns user based on environment mode"""
    if ENV == "dev":
        logger.info("Development mode: skipping authentication")
        return "dev_user"
    else:
        try:
            return verify_token(request)
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            raise HTTPException(status_code=401, detail="Authentication failed")


# Load models
MODELS = {}
try:
    logger.info("Loading models...")
    models_dir = os.path.join(os.path.dirname(__file__), "..", "intent_classifier", "models")
    
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith(".keras")]
        
        for model_file in model_files:
            model_path = os.path.join(models_dir, model_file)
            model_name = model_file.replace(".keras", "")
            
            try:
                MODELS[model_name] = IntentClassifier(load_model=model_path)
                logger.info(f"✅ Model loaded: {model_name}")
            except Exception as e:
                logger.error(f"❌ Failed to load model {model_name}: {str(e)}")
                continue
    else:
        logger.warning(f"Models directory not found: {models_dir}")
    
    if not MODELS:
        logger.error("⚠️  NO MODELS LOADED! API will not work properly.")
    else:
        logger.info(f"Total models loaded: {len(MODELS)}")
        
except Exception as e:
    logger.error(f"Failed during model loading: {str(e)}")
    logger.error(traceback.format_exc())


# Routes
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": f"Basic ML App is running in {ENV} mode",
        "models_loaded": list(MODELS.keys()),
        "database_connected": collection is not None
    }


@app.post("/predict")
async def predict(text: str, owner: str = Depends(conditional_auth)):
    """Make predictions using all loaded models"""
    
    # Check if models are available
    if not MODELS:
        raise HTTPException(
            status_code=503, 
            detail="No models available. Service temporarily unavailable."
        )
    
    # Generate predictions
    predictions = {}
    for model_name, model in MODELS.items():
        try:
            top_intent, all_probs = model.predict(text)
            predictions[model_name] = {
                "top_intent": top_intent,
                "all_probs": all_probs
            }
        except Exception as e:
            logger.error(f"Prediction failed for model {model_name}: {str(e)}")
            predictions[model_name] = {
                "error": str(e)
            }
    
    results = {
        "text": text,
        "owner": owner,
        "predictions": predictions,
        "timestamp": int(datetime.now(timezone.utc).timestamp())
    }
    
    # Try to save to database (if connected)
    if collection is not None:
        try:
            insert_result = collection.insert_one(results.copy())
            results['id'] = str(insert_result.inserted_id)
            logger.info(f"Saved prediction to database: {results['id']}")
        except Exception as e:
            logger.error(f"Failed to save to database: {str(e)}")
            results['id'] = None
            # Continue mesmo se falhar ao salvar no banco
    else:
        results['id'] = None
        logger.warning("Database not connected, prediction not saved")
    
    return JSONResponse(content=results)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all uncaught exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)