# Semantic Similarity API

API de FastAPI que mide la similitud semántica entre dos frases usando `all-MiniLM-L6-v2`.

## Endpoint

`POST /similarity`

```json
{
  "sentence1": "Hoy hace calor en Santiago",
  "sentence2": "La capital chilena está con altas temperaturas"
}
