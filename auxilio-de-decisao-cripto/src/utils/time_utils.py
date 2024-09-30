from datetime import datetime

def convert_timestamp_to_datetime(timestamp: float):
    """Converte um timestamp para um objeto datetime."""
    return datetime.fromtimestamp(timestamp)

def format_datetime(dt: datetime, fmt: str = "%Y-%m-%d %H:%M:%S"):
    """Formata um objeto datetime em uma string."""
    return dt.strftime(fmt)

def calculate_time_diff(start_time: datetime, end_time: datetime):
    """Calcula a diferença de tempo entre dois objetos datetime."""
    return (end_time - start_time).total_seconds()
