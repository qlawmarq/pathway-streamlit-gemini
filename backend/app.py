#!/usr/bin/env python3
"""
RAG Application with Pathway and Gemini
Pydantic Settings + YAML Configuration Management
"""

import os
import sys
import glob
import pathway as pw
from pathlib import Path
from typing import List
import logging

from pathway.xpacks.llm import llms, embedders, parsers, splitters
from pathway.xpacks.llm.vector_store import VectorStoreServer
from pathway.xpacks.llm.question_answering import BaseRAGQuestionAnswerer
from pathway.xpacks.llm.servers import QARestServer
from pathway.udfs import DiskCache, ExponentialBackoffRetryStrategy

from settings import get_settings


def setup_logger(name: str = "rag_app", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)
    if not logger.hasHandlers():
        logger.addHandler(handler)
    return logger


logger = setup_logger()


def setup_llm() -> llms.LiteLLMChat:
    settings = get_settings()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY not found in environment variables")
        sys.exit(1)

    logger.info(f"Connecting to {settings.llm.model_name}...")

    return llms.LiteLLMChat(
        model=settings.llm.model_name,
        api_key=api_key,
        temperature=settings.llm.temperature,
        max_tokens=settings.llm.max_tokens,
        retry_strategy=ExponentialBackoffRetryStrategy(
            max_retries=settings.llm.max_retries
        ),
        cache_strategy=DiskCache(),
    )


def setup_components() -> tuple:
    settings = get_settings()

    parser = parsers.UnstructuredParser()
    embedder = embedders.SentenceTransformerEmbedder(
        model=settings.embedding.model_name
    )
    splitter = splitters.TokenCountSplitter(
        max_tokens=settings.text_processing.max_tokens_per_chunk
    )

    return parser, embedder, splitter


def get_filtered_files(data_path: str) -> List[str]:
    settings = get_settings()
    file_config = settings.data_source.file_filtering

    logger.info(f"Scanning files in {data_path}...")

    all_files = []
    for root, dirs, files in os.walk(data_path):
        if ".git" in dirs:
            dirs.remove(".git")
        if "__pycache__" in dirs:
            dirs.remove("__pycache__")

        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)

    filtered_files = []
    for file_path in all_files:
        should_exclude = False
        relative_path = os.path.relpath(file_path, data_path)

        for pattern in file_config.excluded_patterns:
            if Path(file_path).match(pattern) or glob.fnmatch.fnmatch(
                relative_path, pattern.lstrip("*/")
            ):
                should_exclude = True
                break

        if not should_exclude:
            ext = os.path.splitext(file_path)[1].lower()
            filename = os.path.basename(file_path)

            if (
                ext in file_config.allowed_extensions
                or filename in file_config.allowed_filenames
            ):
                filtered_files.append(file_path)

    logger.info(
        f"Found {len(all_files)} total files, filtered to {len(filtered_files)} files"
    )

    if settings.development.show_file_list and len(filtered_files) > 0:
        logger.debug("Filtered files:")
        for file_path in sorted(filtered_files):
            logger.debug(f"   - {os.path.relpath(file_path, data_path)}")

    return filtered_files


def setup_data_source() -> pw.Table:
    settings = get_settings()
    data_path = settings.data_source.path

    if not os.path.exists(data_path):
        logger.error(f"Data directory not found: {data_path}")
        sys.exit(1)

    logger.info(f"Loading documents from {data_path}")

    filtered_files = get_filtered_files(data_path)

    if not filtered_files:
        logger.error("No valid files found after filtering")
        sys.exit(1)

    tables = []
    for file_path in filtered_files:
        try:
            table = pw.io.fs.read(
                path=file_path,
                format="binary",
                mode=settings.data_source.mode,
                with_metadata=True,
            )
            tables.append(table)
        except Exception as e:
            logger.warning(f"Skipping file {file_path}: {e}")
            continue

    if not tables:
        logger.error("Failed to load any files")
        sys.exit(1)

    if len(tables) > 1:
        logger.info("Resolving table universe conflicts...")
        pw.universes.promise_are_pairwise_disjoint(*tables)

    combined_table = tables[0]
    for table in tables[1:]:
        combined_table = combined_table.concat(table)

    return combined_table


def setup_rag(
    llm: llms.LiteLLMChat, vector_server: VectorStoreServer
) -> BaseRAGQuestionAnswerer:
    settings = get_settings()

    return BaseRAGQuestionAnswerer(
        llm=llm,
        indexer=vector_server,
        search_topk=settings.rag.search_topk,
        prompt_template=settings.rag.prompt_template,
    )


def main():
    logger.info("Starting RAG Application...")

    settings = get_settings()

    if settings.development.debug_logging:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    logger.info("Setting up components...")
    llm = setup_llm()
    parser, embedder, splitter = setup_components()
    data_source = setup_data_source()

    logger.info("Initializing vector store...")
    vector_server = VectorStoreServer(
        data_source, embedder=embedder, splitter=splitter, parser=parser
    )

    logger.info("Setting up RAG system...")
    rag = setup_rag(llm, vector_server)

    logger.info(f"Starting server on {settings.server.host}:{settings.server.port}")
    logger.info("Available endpoints:")
    logger.info("   - POST /v1/pw_ai_answer - Question answering")
    logger.info("   - GET  /v1/statistics   - System statistics")
    logger.info("   - POST /v1/pw_list_documents - List documents")

    server = QARestServer(
        host=settings.server.host, port=settings.server.port, rag_question_answerer=rag
    )

    server.run(with_cache=settings.server.with_cache)


if __name__ == "__main__":
    main()
