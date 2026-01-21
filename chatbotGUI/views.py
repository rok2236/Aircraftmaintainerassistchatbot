# django는 5.1 버전을 사용했습니다.
# 사용자의 질문을 모델에게 전달하고 generate한 결과를 사용자에게 전달하는 코드

from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from chatbotGUI.generater import models_run
import json
from django.shortcuts import render
import uuid

RESPONSE_STORE = {}

def generate_response_id():
    return str(uuid.uuid4())

@csrf_exempt


def chat_response_post(request):
    """
    Step 1: Receive the user's message via a POST request, call models_run, 
    and return the response_message and response_id.
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_message = data.get('message', '')

            response_message = models_run([user_message])
            response_id = generate_response_id()
            RESPONSE_STORE[response_id] = response_message

            return JsonResponse({'response_id': response_id})
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)

    return JsonResponse({'error': 'Invalid request method'}, status=400)


def chat_response_stream(request):
    """
    Step 2: The frontend accesses the interface using EventSource via a GET request and provides the response_id,
    while the backend streams the stored response_message using SSE.
    """
    if request.method == 'GET':
        response_id = request.GET.get('response_id', '')
        response_message = RESPONSE_STORE.get(response_id)
        if not response_message:
            return JsonResponse({'error': 'Invalid response_id or no data found'}, status=400)
        
        # 모델의 답을 한글자씩 보여주는 코드
        def event_stream():
            for char in response_message:
                # print(char)
                yield f"data: {char}\n\n"
            yield "data: __end__\n\n"

        response = StreamingHttpResponse(event_stream(), content_type='text/event-stream')
        response['Cache-Control'] = 'no-cache'
        return response

    return JsonResponse({'error': 'Invalid request method'}, status=400)


def index(request):
    return render(request, 'index.html')

