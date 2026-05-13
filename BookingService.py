import csv
from typing import Dict, List, TypedDict
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.graph import StateGraph
from langchain.agents import create_agent



# -----------------------------------------------------------------------------
# BnBBookingService (Mock REST API client)
# -----------------------------------------------------------------------------

class BnBOffer(TypedDict): #A
    bnb_id: int
    bnb_name: str
    town: str
    available_rooms: int
    price_per_room: float


class BnBBookingService:

    llm_model: ChatOpenAI
    hotel_db_toolkit_tools: SQLDatabaseToolkit
    mock_bnb_offers: List[BnBOffer] = []

    @staticmethod
    def initialize(model: ChatOpenAI):
        BnBBookingService.llm_model = model
        hotel_db = SQLDatabase.from_uri("sqlite:///cornwall_hotels.db")
        hotel_db_toolkit = SQLDatabaseToolkit(db=hotel_db, llm=BnBBookingService.llm_model)
        BnBBookingService.hotel_db_toolkit_tools = hotel_db_toolkit.get_tools()

        with open("manning_works\\ai_agents_apps_langchain_graphs\\travel_infor_agents_app\\cornwall_bnbs.csv") as f:
            mock_bnb_offers = list(csv.DictReader(f))

        for b in mock_bnb_offers:
            b["bnb_id"] = int(b["bnb_id"])
            b["available_rooms"] = int(b["available_rooms"])
            b["price_per_room"] = float(b["price_per_room"])

        BnBBookingService.mock_bnb_offers = mock_bnb_offers


    @tool(description="""Get the SQL database toolkit for booking purposes.""") #A
    @staticmethod
    def get_booking_toolkit() -> SQLDatabaseToolkit:
        return BnBBookingService.hotel_db_toolkit_tools
    

    @staticmethod
    def get_offers_near_town(town: str, num_rooms: int) -> List[BnBOffer]:
        # Mocked REST API response: multiple BnBs per destination
        mock_bnb_offers = BnBBookingService.mock_bnb_offers

        offers = [offer for offer in 
            mock_bnb_offers 
            if offer["town"].lower() == town.lower() 
               and offer["available_rooms"] >= num_rooms]
        return offers
    

    @tool(description="""Check BnB room availability and price for a destination in Cornwall.""") #A
    @staticmethod
    def check_bnb_availability( destination: str, num_rooms: int) -> List[Dict]: #B

        offers = BnBBookingService.get_offers_near_town(destination, num_rooms)
        if not offers:
            return [{"error": f"No available BnBs found in {destination} for {num_rooms} rooms."}]
        return offers
    
    @staticmethod
    def get_accommodation_booking_agent(agent_state) -> StateGraph:
        # -----------------------------------------------------------------------------
        # Accommodation Booking Agent
        # -----------------------------------------------------------------------------
        BOOKING_TOOLS = [BnBBookingService.get_booking_toolkit, BnBBookingService.check_bnb_availability] 
        
        return create_agent (
            model=BnBBookingService.llm_model,
            tools=BOOKING_TOOLS,
            state_schema=agent_state,
            system_prompt="""You are a helpful assistant that can check 
            hotel and BnB room availability and price for a destination in Cornwall. You can use the tools to 
            get the information you need. If the users does not specify the accommodation type, you should 
            check both hotels and BnBs.""",
        )




