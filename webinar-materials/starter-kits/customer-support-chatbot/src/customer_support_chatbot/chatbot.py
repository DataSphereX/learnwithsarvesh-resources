from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI


def _format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


@dataclass
class FAQChatbot:
    api_key: str
    model_name: str = "gpt-4o-mini"
    history_store: dict[str, InMemoryChatMessageHistory] = field(default_factory=dict)

    def get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in self.history_store:
            self.history_store[session_id] = InMemoryChatMessageHistory()
        return self.history_store[session_id]

    def build_chain(self, retriever):
        llm = ChatOpenAI(model=self.model_name, temperature=0, api_key=self.api_key)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a customer support assistant. Use FAQ context to answer accurately. "
                    "If the answer is not in context, say you are not sure.",
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "Context:\n{context}\n\nQuestion: {question}"),
            ]
        )

        base_chain = (
            RunnableParallel(
                context=lambda x: _format_docs(retriever.invoke(x["question"])),
                question=lambda x: x["question"],
                history=lambda x: x.get("history", []),
            )
            | prompt
            | llm
            | StrOutputParser()
        )

        return RunnableWithMessageHistory(
            base_chain,
            self.get_session_history,
            input_messages_key="question",
            history_messages_key="history",
        )

    def ask(self, qa_chain, question: str, session_id: str) -> str:
        config = {"configurable": {"session_id": session_id}}
        return qa_chain.invoke({"question": question}, config=config)
