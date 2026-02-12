# gui/solapi_service.py
import mysql.connector
from sdk.message import Message
from gui.database_models import config # 기존 DB 설정 활용

def get_solapi_config():
    """DB에서 솔라피 설정을 로드합니다."""
    conn = mysql.connector.connect(**config)
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT setting_key, setting_value FROM system_settings WHERE setting_key LIKE 'SOLAPI_%' OR setting_key = 'EMERGENCY_CONTACT'")
    settings = {row['setting_key']: row['setting_value'] for row in cursor.fetchall()}
    conn.close()
    return settings

def send_emergency_alert(text: str):
    """긴급 호출 시 솔라피를 통해 메시지를 발송합니다."""
    settings = get_solapi_config()
    
    # 필수 설정값 확인
    api_key = settings.get('SOLAPI_API_KEY')
    api_secret = settings.get('SOLAPI_API_SECRET')
    to_number = settings.get('EMERGENCY_CONTACT')
    
    if not all([api_key, api_secret, to_number]):
        print("솔라피 설정이 미비합니다. DB를 확인하세요.")
        return False

    message = Message(api_key, api_secret)
    data = {
        'to': to_number,
        'from': '발신번호등록필요', # 솔라피 콘솔에 등록된 번호
        'text': f"[Home Safe 긴급알림] {text}",
        'type': 'SMS' # 알림톡 템플릿 미등록 시 SMS 우선 사용
    }
    
    try:
        response = message.send_one(data)
        return response
    except Exception as e:
        # 에러 발생 시 로그 기록
        return None