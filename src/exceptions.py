import sys
from src.logs import logging
def error_message_details(error, error_Details:sys):
    _,_,exc_tb=error_Details.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename #learnt from Custom Eception Handeling 
    error_messsage= "You have an Error in Script- [{0}] at line- [{1}] || Error Message- [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    
    )
    return error_messsage

class Customed_exception(Exception):
    def __init__(self, error_message, error_Details:sys):
        super().__init__(error_message)
        self.error_message =error_message_details(error_message, error_Details= error_Details)

    def __str__(self):
        return self.error_message


