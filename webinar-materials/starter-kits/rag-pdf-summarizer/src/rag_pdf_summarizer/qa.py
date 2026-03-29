from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def _format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


@dataclass
class PDFRAG:
    api_key: str
    model_name: str = "gpt-4o-mini"

    def build_chain(self, vectorstore_dir: Path):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=self.api_key)
        vectorstore = FAISS.load_local(
            str(vectorstore_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        llm = ChatOpenAI(model=self.model_name, temperature=0, api_key=self.api_key)

        prompt = ChatPromptTemplate.from_template(
            """
Use the context to answer the user question.
If context does not contain the answer, clearly say you are not sure.

Context:
{context}

Question:
{input}
""".strip()
        )

        rag_chain = (
            RunnableParallel(
                context=lambda x: _format_docs(retriever.invoke(x["input"])),
                input=lambda x: x["input"],
            )
            | prompt
            | llm
            | StrOutputParser()
        )
        return rag_chain
