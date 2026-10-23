import os
import urllib.request

# Diccionario con URLs actualizadas y estables
files = {
  "typer-readme.md": "https://raw.githubusercontent.com/tiangolo/typer/main/README.md",
  "fastapi-readme.md": "https://raw.githubusercontent.com/fastapi/fastapi/master/README.md",
  "requests-contributing.md": "https://raw.githubusercontent.com/psf/requests/main/CONTRIBUTING.md",
  "qdrant-readme.md": "https://raw.githubusercontent.com/qdrant/qdrant/master/README.md",
  "openapi-generator-markdown-generator.md": "https://raw.githubusercontent.com/OpenAPITools/openapi-generator/master/docs/generators/markdown.md",
  "langchain-readme.md": "https://raw.githubusercontent.com/langchain-ai/langchain/master/README.md",
  "kubernetes-debug-pods.md": "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/tasks/debug/debug-application/debug-running-pod.md",
  "flask-readme.md": "https://raw.githubusercontent.com/pallets/flask/main/README.md",
  "system-design-primer.md": "https://raw.githubusercontent.com/donnemartin/system-design-primer/master/README.md",
  "pydantic-readme.md": "https://raw.githubusercontent.com/pydantic/pydantic/main/README.md",
}

# The golden-dataset folder next to this script (git-ignored).
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(output_dir, exist_ok=True)

print("Iniciando la descarga del dataset de prueba corregido...")
for filename, url in files.items():
  filepath = os.path.join(output_dir, filename)
  try:
    urllib.request.urlretrieve(url, filepath)
    print(f"[OK] Descargado: {filename}")
  except Exception as e:
    print(f"[ERROR] No se pudo descargar {filename}: {e}")

print(
  f"\n¡Listo! Todos los ficheros disponibles se han guardado en la carpeta '{output_dir}/'."
)
