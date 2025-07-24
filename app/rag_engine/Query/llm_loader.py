#!/usr/bin/env python
# -*- coding: utf-8 -*-

from langchain_ollama import OllamaLLM

def get_llm():
    return OllamaLLM(model="gemma3:1b-it-qat")