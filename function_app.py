import azure.functions as func
import logging
from function_1 import pipeline_lyrae_talk  # Import your existing function

app = func.FunctionApp()

@app.function_name(name="EventGridTrigger")
@app.event_grid_trigger(arg_name="event")
def event_grid_trigger(event: func.EventGridEvent):
    logging.info('Python EventGrid trigger function processed an event')
    pipeline_lyrae_talk()