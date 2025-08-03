import asyncio
import os
from contextlib import asynccontextmanager
from typing import Dict
import time
import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import RedirectResponse
from loguru import logger
from pipecat_ai_small_webrtc_prebuilt.frontend import SmallWebRTCPrebuiltUI
from datetime import datetime
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
from pipecat.transports.network.webrtc_connection import IceServer, SmallWebRTCConnection
# from pipecat.services.nim.llm import NimLLMService
from pipecat.services.openai.llm import OpenAILLMService
# from pipecat.services.riva.stt import RivaSTTService
from pipecat.services.cartesia.stt import CartesiaSTTService
# from pipecat.services.riva.tts import RivaTTSService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.transcriptions.language import Language
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.utils.text.markdown_text_filter import MarkdownTextFilter
from typing import TypedDict, Union, List
from pipecat_flows import (
    FlowArgs,
    FlowManager,
    FlowsFunctionSchema,
    NodeConfig,
    FlowResult
)
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from datetime import date
from dateutil.relativedelta import relativedelta
from num2words import num2words
from pipecat.frames.frames import EndFrame
from pipecat.frames.frames import TTSSpeakFrame


load_dotenv(override=True)

app = FastAPI()

pcs_map: Dict[str, SmallWebRTCConnection] = {}

ice_servers = [
    IceServer(
        urls="stun:stun.l.google.com:19302",
    )
]

app.mount("/client", SmallWebRTCPrebuiltUI)

class PaymentData(TypedDict):
    six_monthly_installment: float
    start_date: str
    six_end_date: str
    twelve_monthly_installment: float
    twelve_end_date: str

class CustomerRecord(FlowResult):
    name: str
    due_amount: float
    due_date: str
    hsa_bal: float

class SetupPaymentPlan(FlowResult):
    user_response: bool

class TypeofPaymentPlan(FlowResult):
    plan_type: str


class PaymentPlanDetails(FlowResult, PaymentData):
    pass

def ordinal_day(day: int) -> str:
    return 'th' if 11 <= day <= 13 else {1:'st', 2:'nd', 3:'rd'}.get(day % 10, 'th')

def spoken_amount(amount: float) -> str:
    words = num2words(amount, to='cardinal', lang='en')
    return f"{words} dollars"

def spoken_date(iso_date_str: str) -> str:
    # Converts ISO date to spoken phrase like:
    # "July thirty-first, twenty twenty-five"
    dt = datetime.date.fromisoformat(iso_date_str)
    day = dt.day
    month = dt.strftime("%B")        # e.g. "July"
    
    # Get day in ordinal words
    day_words = num2words(day, to='ordinal', lang='en')
    
    # Format year as calendar e.g. "2025" -> "twenty twenty-five"
    year_words = num2words(dt.year, to='year', lang='en')
    
    return f"{month} {day_words}, {year_words}"


async def fetch_customer_records_handler(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[CustomerRecord, NodeConfig]:
    name = args["name"]
    sample: List[CustomerRecord] = [
        {"name": "Henry Smith", "due_amount": 5500.00, "due_date": "2025-07-15", "hsa_bal": 500.00},
        {"name": "Jane Doe",   "due_amount": 3200.00, "due_date": "2025-07-20", "hsa_bal": 200.00},
        {"name": "Aditya Gupta",   "due_amount": 200.00, "due_date": "2025-07-30", "hsa_bal": 250.00},
        {"name": "Vivek Joshi",   "due_amount": 3000.00, "due_date": "2025-08-20", "hsa_bal": 125.00},
        {"name": "Abhinav",   "due_amount": 5000.00, "due_date": "2025-08-10", "hsa_bal": 900.00},
    ]
    matches = [r for r in sample if name.lower() in r["name"].lower()]
    flow_manager.state["records"] = matches
    if len(matches) == 1:
        next_node = create_validate_node()
    elif len(matches) > 1:
        next_node = create_multiple_node()
    else:
        next_node = create_no_node()
    return matches, next_node

async def validate_customer_handler(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[CustomerRecord, NodeConfig]:
    record = flow_manager.state["records"][0]
    flow_manager.state["customer"] = record
    # customer_data = CustomerData(**record)
    return record, create_collection_node(record)

async def ask_user_to_setup_plan_handler(args: FlowArgs, flow_manager: FlowManager) -> tuple[PaymentPlanDetails, NodeConfig]:
    user_response = args["user_response"]
    result = SetupPaymentPlan(user_response=user_response)
    if str(user_response).lower() == 'true':
        due_amount = flow_manager.state["customer"].get('due_amount')
        six_monthly_installment = round(due_amount/6, 2)
        start_date = date.today()
        six_end_date = start_date + relativedelta(months=6)
        twelve_monthly_installment = round(due_amount/12, 2)
        twelve_end_date = start_date + relativedelta(months=12)
        result = {
            'six_monthly_installment': six_monthly_installment,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'six_end_date': six_end_date.strftime('%Y-%m-%d'),
            'twelve_monthly_installment': twelve_monthly_installment,
            'twelve_end_date': twelve_end_date.strftime('%Y-%m-%d')
        }
        flow_manager.state["plan_type"] = result
        next_node = create_plan_discussion_node(result)
    else:
        result = {
            'six_monthly_installment': None,
            'start_date': None,
            'six_end_date': None,
            'twelve_monthly_installment': None,
            'twelve_end_date': None
        }
        next_node = end_node()

    return result, next_node

async def setup_payment_plan_handler(args: FlowArgs, flow_manager: FlowManager) -> tuple[PaymentPlanDetails, NodeConfig]:
    plan_type = args["plan_type"]
    due_amount = flow_manager.state["customer"].get('due_amount')
    if plan_type.lower() == 'six_months':
        # plan_type = 'six_months'
        monthly_installment = round(due_amount/6, 2)
        start_date = date.today()
        end_date = start_date + relativedelta(months=6)
        result = {
            'monthly_installment': monthly_installment,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d')
        }
        # flow_manager.state["plan_type"] = result
        next_node = six_month_setup_node()
    elif plan_type.lower() == 'twelve_months':
        # plan_type = 'twelve_months'
        monthly_installment = round(due_amount/12, 2)
        start_date = date.today()
        end_date = start_date + relativedelta(months=12)
        result = {
            'monthly_installment': monthly_installment,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d')
        }
        # flow_manager.state["plan_type"] = result
        next_node = twelve_month_setup_node()
    else:
        result = {
            'monthly_installment': None,
            'start_date': None,
            'end_date': None
        }
        flow_manager.state["plan_type"] = result
        # time.sleep(10)
        next_node = end_node()

    return result, next_node

# async def intermediate_end_conversation_handler(args: FlowArgs) -> tuple[FlowResult, NodeConfig]:
#     logger.debug("intermediate_end_conversation_handler executing")
#     result = {"status": "completed"}
#     next_node = end_node()
#     return result, next_node

async def graceful_end_handler(args: FlowArgs, flow_manager: FlowManager):
    """Handler that ensures TTS completes before ending"""
    await flow_manager.task.queue_frame(TTSSpeakFrame("Have a nice day!"))
    await flow_manager.task.queue_frame(EndFrame())

async def end_conv(args: FlowArgs) -> tuple[FlowResult, NodeConfig]:
    # time.sleep(20)
    logger.debug("end_conv handler executing")
    result = {"status": "completed"}
    next_node = end_node()
    return result, next_node

def create_initial_node() -> NodeConfig:
    """Create the initial node asking for age."""
    return {
        "name": "initial",
        "role_messages": [
            {
                "role": "system",
                "content": (
                    "You are Joanna, an empathetic voice-based debt collection assistant. "
                    "Calls are recorded. Speak naturally with human-like pauses, "
                    "silently call any functions never mention them aloud. "
                    "and there gonna be some instructions present in user section follow those instruction properly without mentioning them aloud. "
                    "Your output will be converted to audio so don't include special characters in your answers. "
                    "Always keep the generation of text short like 2 - 3 senctences only."
                ),
            }
        ],
        "task_messages": [
            {
                "role": "user",
                "content": (
                    "[INSTRUCTION: Start the conversation with introducing yourself first]"
                    "Hello! I'm Joanna from abc. I'll guide you through your account today. May I have your full name, please?"),
            },
        ],
        "functions": [
            FlowsFunctionSchema(
                name="fetch_customer_records",
                description="Lookup customer records by name",
                properties={"name": {"type": "string"}},
                required=["name"],
                handler=fetch_customer_records_handler,
            )
        ],
    }


def create_ask_name_node() -> NodeConfig:
    """Create node for Asking Name."""
    return {
        "name": "ask_name",
        "task_messages": [
            {
                "role": "system",
                "content": (
                    "[INSTRUCTION: Ask the user for their name. You are already in the middle of the conversation, make sure generated text will feel like natural and continuous part of conversation, without using words like 'Certainly'.]"
                ),
            }
        ],
        "functions": [
            FlowsFunctionSchema(
                name="fetch_customer_records",
                description="Lookup customer records by name",
                properties={"name": {"type": "string"}},
                required=["name"],
                handler=fetch_customer_records_handler,
            )
        ],
    }


def create_multiple_node() -> NodeConfig:
    return {
        "name": "multiple_matches",
        "task_messages": [
            {
                "role": "system",
                "content": (
                    "[INSTRUCTION: If multiple records are found for a name, ask the user for their full name. You are already in the middle of the conversation, make sure generated text will feel like natural and continuous part of conversation, without using words like 'Certainly'.]"
                ),
            }
        ],
        "functions": [
            FlowsFunctionSchema(
                name="fetch_customer_records",
                description="Lookup customer records by name",
                properties={"name": {"type": "string"}},
                required=["name"],
                handler=fetch_customer_records_handler,
            )
        ],
    }


def create_no_node() -> NodeConfig:
    return {
        "name": "no_matches",
        "task_messages": [
            {
                "role": "system",
                "content": (
                    "[INSTRUCTION: If no matching record is found, then inform the user that there no records avaialble with this name. And ask for name again. You are already in the middle of the conversation, make sure generated text will feel like natural and continuous part of conversation, without using words like 'Certainly'.]"
                ),
            }
        ],
        "functions": [
            FlowsFunctionSchema(
                name="fetch_customer_records",
                description="Lookup customer records by name",
                properties={"name": {"type": "string"}},
                required=["name"],
                handler=fetch_customer_records_handler,
            )
        ],
    }


def create_validate_node() -> NodeConfig:
    return {
        "name": "validate_record",
        "task_messages": [
            {
                "role": "system",
                "content": (
                    "[INSTRUCTION: Validate the information from the record. You are already in the middle of the conversation, make sure generated text will feel like natural and continuous part of conversation, without using words like 'Certainly'.]"
                ),
            }
        ],
        "functions": [
                    FlowsFunctionSchema(
                        name="validate_customer",
                        description="Confirm exact customer match",
                        properties={},
                        required=[],
                        handler=validate_customer_handler,
                    )
                ],
    }

def create_collection_node(customer: dict) -> NodeConfig:
    # due_amt = f"{customer['due_amount']:.2f}"
    due_amt_words = spoken_amount(customer['due_amount'])
    try:
        due_date = datetime.date.fromisoformat(customer["due_date"])
        due_date_str = due_date.strftime("%B %d, %Y")
    except Exception:
        due_date_str = customer["due_date"]
    return {
        "name": "collection_flow",
        "pre_actions": [
            {
                "type": "tts_say",
                "text": (
                    f"Hi {customer['name']}. "
                    f"You have {due_amt_words} "
                    f"due on {due_date_str}. "
                    "Would you like to set up a payment plan today?"
                ),
            }
        ],
        "respond_immediately": False,
        "task_messages": [
            {
                "role": "system",
                 "content": (
                    "The user has been informed about their due amount and date above. "
                    "If they say yes, call ask_user_to_setup_plan with user_response=true. "
                    "If they say no or ask for something else, respond accordingly and do not call the function."
                ),
            }
        ],
        "functions": [
                    FlowsFunctionSchema(
                        name="ask_user_to_setup_plan",
                        description="If user like to setup a payment plan then it's a True otherwise it's a False.",
                        properties={"user_response": {"type": "boolean"}},
                        required=["user_response"],
                        handler=ask_user_to_setup_plan_handler,
                    )
                ],
    }

def create_plan_discussion_node(plan: PaymentPlanDetails) -> NodeConfig:
    return {
        "name": "plan_discussion",
        "task_messages": [
            {
                "role": "system",
                "content": (
                    "[INSTRUCTION: Propose the customer with six month payment plan and twelve month payment plan in detail as shown in example below and follow the plan selection flow. User can also deny both of these plans.] "
                    "You are already in the middle of the conversation, make sure generated text will feel like natural and continuous part of conversation, without using words like 'Certainly'.] " 
                    "I can offer a six months payment plan at {{plan.six_monthly_installment | to_words}} dollars per month from {{plan.start_date | format_date('%B %d, %Y')}} until {{plan.six_end_date | format_date('%B %d, %Y')}}. "
                    "Does that work for you? "
                    "Or twelve months plan at {{plan.twelve_monthly_installment | to_words}} dollars per month from {{plan.start_date | format_date('%B %d, %Y')}} until {{plan.twelve_end_date | format_date('%B %d, %Y')}}? "
                    "[Based on the user's response, call the setup_payment_plan function with the selected plan type or deny option.]"
                )
            }
        ],
        "functions": [
                    FlowsFunctionSchema(
                        name="setup_payment_plan",
                        description="Choose type of payment plan.",
                        properties={"plan_type": {"type": "string", "enum": ["six_months", "twelve_months", "deny"]}},
                        required=["plan_type"],
                        handler=setup_payment_plan_handler,
                    )
                ],
    }

def six_month_setup_node() -> NodeConfig:
    return {
        "name": "six_month_accept",
        "task_messages": [{
                "role": "system",
                "content": (
                    "[INSTRUCTION: Inform the customer about setting up of six months plan and ask to use HSA balance towards the payment. You are already in the middle of the conversation, make sure generated text will feel like natural and continuous part of conversation, without using words like 'Certainly'.] "
                    "Great! I have set up your six months plan. Would you like to use your HSA balance of {{customer.hsa_bal | to_words}} dollars towards this payment? "
                    "If customer says yes then inform them that their due amount will be updated and ask for confirmation. If they say no then inform them that their due amount will remain same and ask for confirmation. "
                    "If customer agrees then call then greet them and call end conversation to complete the payment plan setup and call end_conversation function." 
                )
            }
        ],
        "functions": [
                    FlowsFunctionSchema(
                        name="end_conversation",
                        description="Complete the payment plan setup.",
                        properties={},
                        required=[],
                        handler=end_conv,
                    )
                ],
    }

def twelve_month_setup_node() -> NodeConfig:
    return {
        "name": "twelve_month_accept",
        "task_messages": [{
                "role": "system",
                "content": (
                    "[INSTRUCTION: Inform the customer about setting up of twelve months plan and ask to use HSA balance towards the payment. You are already in the middle of the conversation, make sure generated text will feel like natural and continuous part of conversation, without using words like 'Certainly'.] "
                    "Great! I have enrolled you in the twelve months plan. Would you like to use your HSA balance of {{customer.hsa_bal | to_words}} dollars towards this payment? "
                    "If customer says yes then inform them that their due amount will be updated and ask for confirmation. If they say no then inform them that their due amount will remain same and ask for confirmation. "
                    "If customer agrees then call then greet them and call end conversation to complete the payment plan setup and call end_conversation function."  
                )
            }
        ],
        "functions": [
                    FlowsFunctionSchema(
                        name="end_conversation",
                        description="Complete the payment plan setup.",
                        properties={},
                        required=[],
                        handler=end_conv,
                    )
                ],
    }

# def deny_node() -> NodeConfig:
#     return {
#         "name": "all_denied",
#         "task_messages": [{
#                 "role": "assistant",
#                 "content": (
#                     "Thank you for your time. Feel free to reach out at any time. Goodbye."
#                 )
#             }
#         ],
#         "functions": [
#                     FlowsFunctionSchema(
#                         name="end_conversation",
#                         description="Complete the payment plan setup.",
#                         properties={},
#                         required=[],
#                         handler=end_conv,
#                     )
#                 ],
#     }
    
def end_node() -> NodeConfig:
    return {
        "name": "completed",
        # "pre_actions": [
        #     {
        #         "type": "tts_say",
        #         "text": (
        #             "Thank you for your time. Feel free to reach out at any time. Goodbye."
        #         ),
        #     }
        # ],
        # "respond_immediately": False,
        "task_messages": [{
                "role": "system",
                "content": (
                    "Say goodbye to the user with 'Have a nice day!'"
                )
            }
        ],
        "post_actions": [{"type": "function", "handler": graceful_end_handler}],
    }

async def run_example(webrtc_connection: SmallWebRTCConnection):
    logger.info(f"Starting bot")

    # Create a transport using the WebRTC connection
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    # stt = RivaSTTService(
    #     api_key=os.getenv("NVIDIA_API_KEY"),
    #     # model_function_map={
    #     # "function_id": "d3fe9151-442b-4204-a70d-5fcc597fd610",
    #     # "model_name": "parakeet-tdt-0.6b-v2"
    #     # },
    #     params=RivaSTTService.InputParams(
    #     language=Language.EN_US
    #     )
    # )
    
    stt = CartesiaSTTService(
        api_key=os.getenv("CARTESIA_API_KEY")
    )

    llm = OpenAILLMService(
        model="gpt-4.1-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        params=OpenAILLMService.InputParams(
            temperature=0.7,
        )
    )
    # tts = ElevenLabsTTSService(
    #     api_key=os.getenv("ELEVENLABS_API_KEY"),
    #     voice_id=os.getenv("ELEVENLABS_VOICE_ID"),
    #     model="eleven_flash_v2_5",
    #     params=ElevenLabsTTSService.InputParams(
    #         language=Language.EN,
    #         stability=0.7,
    #         similarity_boost=0.8,
    #         style=0.5,
    #         use_speaker_boost=True,
    #         speed=1.1
    #     )
    # )
    
    # tts = RivaTTSService(
    #     api_key=os.getenv("NVIDIA_API_KEY"),
    #     voice_id="Magpie-Multilingual.EN-US.Ray",
    #     params=RivaTTSService.InputParams(
    #         language=Language.EN_US,
    #         quality=20
    #     )
    # )
    
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="bf0a246a-8642-498a-9950-80c35e9276b5",
        model="sonic-2",
        params=CartesiaTTSService.InputParams(
            language=Language.EN,
            speed="normal"
        )
    )

    context = OpenAILLMContext()
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    flow_manager = FlowManager(
        task=task,
        llm=llm,
        context_aggregator=context_aggregator,
        transport=transport,
    )


    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
            logger.info(f"Client connected")
            # Kick off the conversation.
            wait_for_user=False
            await flow_manager.initialize(create_initial_node())

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy"}

@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/client/")


@app.post("/api/offer")
async def offer(request: dict, background_tasks: BackgroundTasks):
    pc_id = request.get("pc_id")

    if pc_id and pc_id in pcs_map:
        pipecat_connection = pcs_map[pc_id]
        logger.info(f"Reusing existing connection for pc_id: {pc_id}")
        await pipecat_connection.renegotiate(
            sdp=request["sdp"],
            type=request["type"],
            restart_pc=request.get("restart_pc", False),
        )
    else:
        pipecat_connection = SmallWebRTCConnection(ice_servers)
        await pipecat_connection.initialize(sdp=request["sdp"], type=request["type"])

        @pipecat_connection.event_handler("closed")
        async def handle_disconnected(webrtc_connection: SmallWebRTCConnection):
            logger.info(f"Discarding peer connection for pc_id: {webrtc_connection.pc_id}")
            pcs_map.pop(webrtc_connection.pc_id, None)

        # Run example function with SmallWebRTC transport arguments.
        background_tasks.add_task(run_example, pipecat_connection)

    answer = pipecat_connection.get_answer()
    # Updating the peer connection inside the map
    pcs_map[answer["pc_id"]] = pipecat_connection

    return answer


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield  # Run app
    coros = [pc.disconnect() for pc in pcs_map.values()]
    await asyncio.gather(*coros)
    pcs_map.clear()


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)