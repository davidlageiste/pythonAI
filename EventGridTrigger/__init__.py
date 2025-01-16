import logging
import azure.functions as func
from ..function_1 import pipeline_lyrae_talk

def main(event: func.EventGridEvent):
    logging.info('Python EventGrid trigger function processed an event')
    
    # Get event data
    result = event.get_json()
    event_type = event.event_type
    
    logging.info(f'Event Type: {event_type}')
    logging.info(f'Event Data: {result}')
    
    try:
        pipeline_lyrae_talk()
        logging.info('Pipeline executed successfully')
    except Exception as e:
        logging.error(f'Error executing pipeline: {str(e)}')