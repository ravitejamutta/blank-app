import requests

class LLMService:
    def __init__(self):
        self.backend_url = "https://stma-of-dev.cs.ms.hcvpc.io/convai119/rag"
        self.headers = {"Content-Type": "application/json"}

    def get_response(self, query: str, model_type: str):
        # Only RAG is supported for now via backend
        if model_type != "RAG":
            return {
                "response": "❌ Fine-tuned model not implemented in backend.",
                "confidence": 0.0,
                "method": "Not available",
                "response_time": 0.0
            }

        payload = {
            "query": query,
            "top_k": 5
        }

        try:
            res = requests.post(self.backend_url, headers=self.headers, json=payload)
            res.raise_for_status()
            data = res.json()

            return {
                "response": data.get("answer", "No answer returned."),
                "confidence": data.get("confidence", 0.0),
                "method": data.get("method", "Unknown"),
                "response_time": data.get("response_time", 0.0),
                "retrieved_docs": data.get("retrieved_docs", [])
            }

        except requests.RequestException as e:
            return {
                "response": f"❌ Error: {str(e)}",
                "confidence": 0.0,
                "method": "Error",
                "response_time": 0.0,
                "retrieved_docs": []
            }
