import logging
import azure.functions as func
from ..function_1 import pipeline_lyrae_talk
from azure.communication.callautomation import CallAutomationClient, CallInvite, IncomingCallContext
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
        self.call_automation_client = None
        self.current_call = None

    def initialize(self):
        try:
            # Initialize the CallAutomationClient
            self.call_automation_client = CallAutomationClient.from_connection_string(self.connection_string)
            logging.info("CallAutomationClient initialized successfully.")
        except Exception as e:
            logging.error(f"Error initializing CallAutomationClient: {str(e)}")
            raise

    def accept_call(self, incoming_call_context):
        try:
            # Ensure the client is initialized
            if not self.call_automation_client:
                self.initialize()

            # Create an IncomingCallContext object
            call_context = IncomingCallContext.from_dict(incoming_call_context)

            # Accept the incoming call
            self.current_call = self.call_automation_client.answer_call(
                incoming_call_context=call_context,
                callback_url="<YOUR_CALLBACK_URL>"
            )

            logging.info(f"Call accepted from {call_context.caller_id}")

            # Call the pipeline function with the call connection
            pipeline_lyrae_talk(self.current_call)

            return {
                'status': 'accepted',
                'callId': str(self.current_call.call_connection_id),
                'caller': call_context.caller_id
            }

        except Exception as e:
            logging.error(f"Error accepting call: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }