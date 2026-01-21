from django.apps import AppConfig

class ChatbotGUIConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "chatbotGUI"

    def ready(self):
        # 모델 초기화 함수 호출
        from .models_loader import initialize_models
        initialize_models()
