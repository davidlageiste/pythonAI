import logging
import azure.functions as func
from ..function_1 import pipeline_lyrae_talk
from azure.communication.calling import (
    CallClient,
    CallAgent,
    IncomingCall,
    VideoStreamRenderer,
    LocalVideoStream
)
from azure.communication.identity import CommunicationIdentityClient

def main(event: func.EventGridEvent):
    logging.info('Python EventGrid trigger function processed an event')

    # Get event data
    result = event.get_json()
    event_type = event.event_type

    logging.info(f'Event Type: {event_type}')
    logging.info(f'Event Data: {result}')

    try:
        handle_incoming_call()
        
        logging.info('Pipeline executed successfully')
    except Exception as e:
        logging.error(f'Error executing pipeline: {str(e)}')


def handle_incoming_call(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Processing incoming call request')

    try:
        # Get the request body
        req_body = req.get_json()
        logging.info(f"Received event: {req_body}")

        # Initialize call handler
        connection_string = "endpoint=https://lyraetalk.france.communication.azure.com/;accesskey=3w3cK83UG45fDt4zOVi4mwSsApOvCbqfZhn1tKFn4TPMp5d5umCYJQQJ99ALACULyCpuAreVAAAAAZCS6qkJ"  # Get from environment variables in production
        call_handler = AzureCallHandler(connection_string)

        # Process the incoming call
        result = call_handler.accept_call(req_body)

        return func.HttpResponse(
            json.dumps(result),
            mimetype="application/json",
            status_code=200
        )

    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )
    
class AzureCallHandler:
    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.call_agent = None
        self.call_client = None
        self.current_call = None

    def initialize(self):
        # Create the identity client to get a token
        identity_client = CommunicationIdentityClient.from_connection_string(self.connection_string)
        user = identity_client.create_user()
        token_response = identity_client.get_token(user, ["voip"])

        # Initialize call client and agent
        self.call_client = CallClient()
        self.call_agent = self.call_client.create_call_agent(token_response.token)

    def accept_call(self, incoming_call_context):
        try:
            # Initialize if not already initialized
            if not self.call_agent:
                self.initialize()

            # Create an IncomingCall object from the context
            incoming_call = IncomingCall(
                caller_id=incoming_call_context.get('callerId'),
                correlation_id=incoming_call_context.get('correlationId')
            )

            # Accept the call
            self.current_call = incoming_call.accept()
            logging.info(f"Call accepted from {incoming_call_context.get('callerId')}")

            # Call the pipeline function
            pipeline_lyrae_talk(self.current_call)

            return {
                'status': 'accepted',
                'callId': str(self.current_call.id),
                'caller': incoming_call_context.get('callerId')
            }

        except Exception as e:
            logging.error(f"Error accepting call: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }