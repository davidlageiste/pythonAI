import azure.functions as func
import logging
from function_1 import pipeline_lyrae_talk  # Import your existing function

app = func.FunctionApp()

@app.function_name(name="EventGridTrigger")
@app.event_grid_trigger(arg_name="event")
def event_grid_trigger(event: func.EventGridEvent):
    logging.info('Python EventGrid trigger function processed an event')
    
    # Get event data
    event_type = event.event_type
    echo(event_type)
    # Call your specific function based on event type or data
    if event_type == "your.specific.event":
        pipeline_lyrae_talk()