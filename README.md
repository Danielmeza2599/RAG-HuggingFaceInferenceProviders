# Chat RAG con PDF

Esta aplicación local permite cargar uno o varios PDF y formular preguntas en español sobre su contenido. Implementa un flujo de Retrieval-Augmented Generation (RAG):

1. Extrae texto por página con PyMuPDF.
2. Divide el texto en fragmentos con solapamiento.
3. Genera embeddings localmente con Sentence Transformers.
4. Busca los fragmentos más relevantes en un índice FAISS.
5. Envía la pregunta y los fragmentos recuperados a Hugging Face Inference Providers.
6. Muestra la respuesta junto con el archivo, la página y la similitud de las fuentes.

Está orientada a una práctica académica del curso MTW410 y funciona como una interfaz Streamlit sin API propia ni base de datos persistente.

## Tecnologías principales

- Python
- Streamlit
- PyMuPDF
- Sentence Transformers
- FAISS CPU
- Hugging Face Inference Providers
- NumPy

Los modelos predeterminados son:

- Embeddings: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Generación: `openai/gpt-oss-20b`

La disponibilidad del modelo generativo puede variar según el proveedor, la cuenta y las cuotas de Hugging Face.

## Requisitos

- macOS, Windows o Linux; las instrucciones siguientes están verificadas principalmente para macOS.
- Python 3.11 o 3.12 recomendado.
- Conexión a Internet para descargar el modelo de embeddings la primera vez.
- Un token de Hugging Face con permisos para Inference Providers.
- PDF que contengan texto seleccionable.

El OCR no está incluido. Los PDF escaneados sin una capa de texto no podrán indexarse.

## Instalación local

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

En Windows PowerShell, activa el entorno con:

```powershell
.venv\Scripts\Activate.ps1
```

Copia la plantilla de configuración:

```bash
cp .env.example .env
```

Después reemplaza `hf_your_token_here` por tu token real en `.env`. También puedes introducir el token directamente en la barra lateral de Streamlit o exportarlo antes de iniciar la aplicación:

```bash
export HF_TOKEN="tu_token"
```

Para iniciar:

```bash
python -m streamlit run app.py
```

Abre la dirección que muestra Streamlit, normalmente `http://localhost:8501`.

Para detener la aplicación usa `Ctrl+C`. Para desactivar el entorno usa `deactivate`.

## Uso

1. Crea un token en <https://huggingface.co/settings/tokens> con permisos para **Inference Providers**.
2. Configura el token en `.env`, mediante `HF_TOKEN` o en la barra lateral.
3. Sube uno o varios PDF con texto seleccionable.
4. Espera a que Streamlit indique que el índice está listo.
5. Selecciona entre 1 y 6 fragmentos recuperados.
6. Realiza una pregunta en el chat.
7. Revisa la respuesta y abre las fuentes recuperadas para ver el archivo, la página y la similitud.

Ejemplo:

```text
¿Cuáles son las unidades temáticas del curso?
```

## Privacidad y datos

- Los embeddings se calculan localmente.
- La pregunta y los fragmentos recuperados se envían al proveedor externo seleccionado.
- Los documentos, fragmentos, embeddings e historial viven en memoria durante la sesión de Streamlit.
- No se recomienda subir documentos confidenciales ni compartir tokens.
- El archivo `.env` está excluido por `.gitignore`; nunca publiques el token real.

## Arquitectura rápida

- `app.py`: interfaz Streamlit, carga de archivos y estado de sesión.
- `rag_core.py`: extracción de PDF, fragmentación, embeddings, búsqueda FAISS y generación.
- `requirements.txt`: dependencias y rangos de versiones.
- `.env.example`: plantilla de configuración versionable.
- `.env`: configuración local con secretos; excluido de Git.
- `Context/`: documentación interna de agentes; excluida de Git.

Para contexto técnico detallado de agentes de IA, consulta `Context/ContextProject-Agents.md` localmente. Esta carpeta no debe incluirse en el repositorio.

## Validación local

No existe una suite de pruebas automatizada registrada en el proyecto. La validación mínima actual es:

```bash
python -m compileall app.py rag_core.py
python -m streamlit run app.py
```

Prueba el flujo con un PDF que contenga texto seleccionable y una pregunta cuya respuesta pueda verificarse dentro del documento.

## Solución de problemas

- `ModuleNotFoundError`: confirma que activaste `.venv` y ejecutaste `pip install -r requirements.txt`.
- `401/403`: verifica el token, sus permisos y la cuota de Hugging Face.
- `404 / model not supported`: selecciona otro modelo instruct disponible en Hugging Face Inference Providers.
- PDF sin texto: necesita OCR, que no está incluido en esta práctica.
- La primera ejecución descarga el modelo de embeddings y requiere conexión.
- Si cambia el modelo y falla la inferencia, revisa su disponibilidad en tu cuenta y reemplaza el identificador.

