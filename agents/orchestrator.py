"""
FasalSetu Orchestrator — Google ADK v1.x
Fixed: conversation memory, better follow-up handling, clearer tool instructions.
"""
import json
import logging
import os
from dotenv import load_dotenv

load_dotenv()
_api_key = os.getenv("GEMINI_API_KEY", "").strip()
if not _api_key:
    raise EnvironmentError(
        "GEMINI_API_KEY is not set.\n"
        "Add it to your .env: GEMINI_API_KEY=your_key_here\n"
        "Get one at: https://aistudio.google.com/app/apikey"
    )
os.environ["GOOGLE_API_KEY"] = _api_key

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.genai import types as genai_types

from compliance.guardrail import check_and_gate, ComplianceViolation
from agents.soil_agent    import predict_npk, get_soil_health_report
from agents.disease_agent import detect_crop_disease, get_disease_info
from agents.market_agent  import get_market_prices, list_supported_crops
from agents.weather_agent import get_weather_forecast, check_spray_conditions, get_farming_weather_advice

logger = logging.getLogger("fasalsetu")

SYSTEM_PROMPT = """You are FasalSetu, an agricultural advisory AI for Indian farmers.

CONVERSATION MEMORY:
- You maintain conversation context across multiple messages in the same session.
- When a farmer asks a follow-up question, refer back to previous tool calls and answers.
- If they say "what about rice?" after asking about wheat, remember the wheat context.
- If they ask "and the weather?" after a soil question, remember the location from the soil query.

TOOLS AVAILABLE:
- predict_npk(soil_conductivity, soil_humidity, soil_pH, soil_temperature, hour, day_of_year): predict NPK levels
- get_soil_health_report(..., crop): full soil report with crop-specific advice
- detect_crop_disease(image_path): identify disease from image
- get_disease_info(disease_name): get treatment for a known disease
- get_market_prices(crop, state, city): current mandi price and MSP comparison
- list_supported_crops(): show all crops with MSP data
- get_weather_forecast(location): current weather
- check_spray_conditions(location): safe/unsafe verdict for spraying
- get_farming_weather_advice(location, crop): weather-based farming actions

FORMATTING RULES:
1. NEVER use markdown. No asterisks, no hyphens for bullets, no pound signs.
2. Write in plain sentences. Use numbered lists (1. 2. 3.) when listing items.
3. Keep responses under 180 words.
4. Start with the answer directly — do not say "I" or "Based on the tool output".

CONTENT RULES:
5. ALWAYS call the tool before answering. Do not guess values.
6. When soil data is provided, call predict_npk or get_soil_health_report with ALL parameters.
7. For weather questions, call check_spray_conditions or get_weather_forecast with the location.
8. For market questions, call get_market_prices with crop, state, and city if provided.
9. State confidence or data source after predictions (e.g. "confidence: MAE ±0.06 units" or "source: openweathermap-live").
10. Never recommend banned pesticides. If asked, name the ban law and suggest alternatives.
11. For sell/hold questions, always state the MSP and compare it to market price.
12. Only suggest KVK consultation for genuinely urgent or complex field issues."""

APP_NAME = "fasalsetu"

_tools = [
    FunctionTool(predict_npk),
    FunctionTool(get_soil_health_report),
    FunctionTool(detect_crop_disease),
    FunctionTool(get_disease_info),
    FunctionTool(get_market_prices),
    FunctionTool(list_supported_crops),
    FunctionTool(get_weather_forecast),
    FunctionTool(check_spray_conditions),
    FunctionTool(get_farming_weather_advice),
]

_agent = Agent(
    name="fasalsetu_orchestrator",
    model="gemini-2.5-flash",
    description="Agricultural advisory agent for Indian farmers",
    instruction=SYSTEM_PROMPT,
    tools=_tools,
)

_session_service = InMemorySessionService()
_runner = Runner(
    agent=_agent,
    app_name=APP_NAME,
    session_service=_session_service,
    auto_create_session=True,
)


def run_query(user_input: str, context: dict = None, session_id: str = "default") -> dict:
    enriched = user_input
    if context:
        enriched += f"\n\nContext: {json.dumps(context, ensure_ascii=False)}"

    message = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=enriched)],
    )

    try:
        parts = []
        for event in _runner.run(
            user_id="farmer",
            session_id=session_id,
            new_message=message,
        ):
            if event.is_final_response() and event.content:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        parts.append(part.text)

        answer = "".join(parts).strip() or "No response generated."
        response = {
            "answer": answer,
            "agent":  "fasalsetu-adk",
            "model":  "gemini-2.5-flash",
            "input":  user_input,
        }
        return check_and_gate("orchestrator", response)

    except ComplianceViolation as e:
        logger.warning("Compliance block: %s", e)
        return {
            "answer":  "That pesticide is banned under the Insecticides Act 1968. Safe alternatives: neem oil 3mL/L, spinosad, or Bacillus thuringiensis. KVK helpline: 1800-180-1551.",
            "blocked": True,
            "reason":  str(e),
        }
    except Exception as e:
        logger.error("Orchestrator error: %s", e, exc_info=True)
        return {
            "answer": f"Error: {str(e)}",
            "agent":  "fasalsetu-adk",
            "input":  user_input,
            "error":  True,
        }
