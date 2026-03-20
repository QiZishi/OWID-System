global stop_training_flag
stop_training_flag = False

def set_stop_training_flag(value):
    global stop_training_flag
    stop_training_flag = value

def get_stop_training_flag():
    global stop_training_flag
    return stop_training_flag
