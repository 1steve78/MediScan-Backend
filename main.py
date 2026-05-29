from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Any
import logging
import traceback

# Setup clean, structured logging for manual verification and testing
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MediScan-Ai-Backend")

app = FastAPI(
    title="MediScan-Ai Clinical Intelligence API",
    description="Hackathon-optimized AI backend services for clinical triage and scan report analysis.",
    version="1.0.0"
)

# CORS configuration to seamlessly accept incoming requests from the frontend client
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- PYDANTIC SCHEMAS ---

class TriageRequest(BaseModel):
    symptoms: str
    language: str = "en"

class TriageResponse(BaseModel):
    classification: str  # low, medium, high, critical
    ai_insight: str
    status: str = "success"

class AnalyzeRequest(BaseModel):
    file_url: str

class AnalyzeResponse(BaseModel):
    diagnosis: str
    prescriptions: List[str]
    summary: List[str]
    confidence_score: float
    status: str = "success"


# --- EXCEPTION HANDLERS (To prevent backend crashes during testing) ---

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Catch-all global exception handler to guarantee that unhandled errors
    return a clean 500 error instead of causing an API crash.
    """
    error_trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    logger.error(f"Global Exception Hook: {str(exc)}\n{error_trace}")
    return {
        "status": "error",
        "detail": "An unexpected server error occurred during analysis.",
        "error_type": type(exc).__name__,
        "message": str(exc)
    }


# --- API ENDPOINTS ---

@app.get("/")
async def root():
    return {
        "app": "MediScan-Ai Clinical Intelligence API",
        "status": "operational",
        "docs_url": "/docs"
    }


@app.post("/api/triage", response_model=TriageResponse, status_code=status.HTTP_200_OK)
async def post_triage(payload: TriageRequest):
    """
    Triage Symptoms Endpoint:
    Parses incoming symptom queries and determines a recommended triage level 
    (low, medium, high, critical) alongside localized clinical AI insights.
    """
    logger.info(f"Received triage request. Symptoms: '{payload.symptoms}' [Lang: {payload.language}]")
    
    try:
        # Simple input validation
        if not payload.symptoms.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Symptom details cannot be empty."
            )

        symptoms_lower = payload.symptoms.lower()
        classification = "low"
        ai_insight = ""

        # Language localization dictionaries for mock responses
        translations = {
            "en": {
                "critical": "CRITICAL: Immediate emergency intervention advised. Please proceed to the nearest trauma unit or call emergency services.",
                "high": "HIGH PRIORITY: Strong indicator of acute condition. We advise scheduling a clinical diagnostic consultation within 24 hours.",
                "medium": "MEDIUM PRIORITY: Non-acute symptomatic patterns detected. Monitor closely and consult a primary care physician if symptoms persist.",
                "low": "LOW PRIORITY: Normal physiological variations or mild systemic reaction. Rest, stay hydrated, and observe."
            },
            "es": {
                "critical": "CRÍTICO: Se recomienda intervención médica de emergencia inmediata. Diríjase a urgencias o llame a emergencias.",
                "high": "ALTA PRIORIDAD: Indicador fuerte de afección aguda. Se aconseja programar una consulta médica en las próximas 24 horas.",
                "medium": "PRIORIDAD MEDIA: Síntomas no agudos detectados. Controle de cerca y consulte a su médico de cabecera si persisten.",
                "low": "BAJA PRIORIDAD: Variaciones fisiológicas normales o reacción sistémica leve. Reposo, hidratación y observación."
            }
        }

        # Select translations (default to English if language is not supported)
        lang = payload.language.lower() if payload.language.lower() in translations else "en"
        text_db = translations[lang]

        # Smart mock logic based on symptom keywords to make the hackathon demo feel alive
        if any(w in symptoms_lower for w in ["chest pain", "shortness of breath", "heart attack", "unconscious", "stroke"]):
            classification = "critical"
            ai_insight = text_db["critical"]
        elif any(w in symptoms_lower for w in ["fever", "fracture", "severe pain", "bleeding", "migraine"]):
            classification = "high"
            ai_insight = text_db["high"]
        elif any(w in symptoms_lower for w in ["cough", "abdominal", "dizzy", "nausea", "rash"]):
            classification = "medium"
            ai_insight = text_db["medium"]
        else:
            classification = "low"
            ai_insight = text_db["low"]

        # Log classified output
        logger.info(f"Triage classification resolved to: {classification.upper()}")

        return TriageResponse(
            classification=classification,
            ai_insight=ai_insight
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error handling triage analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze symptoms."
        )


@app.post("/api/analyze-report", response_model=AnalyzeResponse, status_code=status.HTTP_200_OK)
async def post_analyze_report(payload: AnalyzeRequest):
    """
    Document/Report Analysis Endpoint:
    Parses a reference URL pointing to a radiological scan report or lab sheet
    and extracts structured clinical metadata (mocked).
    """
    logger.info(f"Received scan analysis request. Document URL: '{payload.file_url}'")
    
    try:
        # Simple path/URL validation
        if not payload.file_url.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Document reference URL cannot be empty."
            )

        url_lower = payload.file_url.lower()
        
        # Smart mock data selection based on the file name or type
        if "brain" in url_lower or "mri" in url_lower:
            diagnosis = "Mild cortical atrophy and scattered nonspecific white matter hyperintensities."
            prescriptions = [
                "Donepezil 5mg (1 tab daily at bedtime)",
                "Vitamin B-Complex supplement (daily)",
                "Follow-up Brain MRI (3D FLAIR sequence) in 6 months"
            ]
            summary = [
                "Ventricular configuration and size are normal for demographic age group.",
                "No acute cerebral infarct, intracranial hemorrhage, or major midline shifts.",
                "Mild chronic microvascular ischemic changes detected in periventricular regions.",
                "Clinical correlation is suggested for early cognitive screening patterns."
            ]
            confidence_score = 0.94
        elif "chest" in url_lower or "xray" in url_lower or "lung" in url_lower:
            diagnosis = "Acute left-sided lobar pneumonia with localized pleural effusion."
            prescriptions = [
                "Amoxicillin-Clavulanate 875/125mg (1 tablet orally every 12 hours for 7 days)",
                "Albuterol HFA Inhaler (90mcg/actuation, 2 puffs every 4-6 hours as needed)",
                "Re-check Chest X-ray in 10-14 days to monitor lung clearing"
            ]
            summary = [
                "Localized opacity observed in the left lower pulmonary lobe, highly indicative of consolidation.",
                "Cardiomediastinal silhouette and aortic knob reside within physiological boundaries.",
                "Diaphragmatic arches are clear, with slight flattening on the left hemithorax.",
                "Trachea resides in a central position; structural airways remain patent."
            ]
            confidence_score = 0.97
        else:
            # Default fallback mock response
            diagnosis = "Unremarkable screening findings. No active localized pathology detected."
            prescriptions = [
                "Regular hydration (2.5L daily)",
                "Prophylactic multivitamin treatment"
            ]
            summary = [
                "All scanned biological tissue architectures fall within standard clinical limits.",
                "No signs of focal consolidations, structural deformities, or mass effects.",
                "Normal physiological parameters documented throughout target scanning frames."
            ]
            confidence_score = 0.91

        logger.info(f"Analysis completed successfully. Primary Diagnosis: '{diagnosis}'")

        return AnalyzeResponse(
            diagnosis=diagnosis,
            prescriptions=prescriptions,
            summary=summary,
            confidence_score=confidence_score
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error executing report analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to compile analytical document data."
        )

if __name__ == "__main__":
    import uvicorn
    logger.info("Booting local testing backend server...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
