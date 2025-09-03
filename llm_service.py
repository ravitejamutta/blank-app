import requests

class LLMService:
    def __init__(self):
        self.headers = {"Content-Type": "application/json"}
        self.rag_url = ""
        self.finetuned_url = ""

    def get_response(self, query: str, model_type: str):
        if model_type == "RAG":
            payload = {
                "query": query,
                "top_k": 5
            }
            url = self.rag_url
        elif model_type == "FINETUNED":
            payload = {
                "question": query
            }
            url = self.finetuned_url
        else:
            return {
                "response": "❌ Unsupported model type.",
                "confidence": 0.0,
                "method": "Not available",
                "response_time": 0.0
            }

        try:
            res = requests.post(url, headers=self.headers, json=payload)
            res.raise_for_status()
            data = res.json()

            return {
                "response": data.get("answer", "No answer returned."),
                "confidence": data.get("confidence", 0.0),
                "method": data.get("method", "Unknown"),
                "response_time": data.get("response_time", 0.0),
                "retrieved_docs": data.get("retrieved_docs", [])  # May not exist in finetuned case
            }

        except requests.RequestException as e:
            return {
                "response": f"❌ Error: {str(e)}",
                "confidence": 0.0,
                "method": "Error",
                "response_time": 0.0,
                "retrieved_docs": []
            }
