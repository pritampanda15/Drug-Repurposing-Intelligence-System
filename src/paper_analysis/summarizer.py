"""
Scientific paper summarization using LLMs.
"""

import logging
import os
from typing import Dict, Optional

try:
    import openai
    import httpx
    import certifi
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

logger = logging.getLogger(__name__)


class PaperSummarizer:
    """Generate summaries of scientific papers."""

    def __init__(self, model: str = None):
        """
        Initialize paper summarizer.

        Parameters
        ----------
        model : str
            OpenAI model to use (defaults to OPENAI_MODEL env var or gpt-3.5-turbo)
        """
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

        if not HAS_OPENAI:
            logger.warning("OpenAI not installed. Summarization will not be available.")
            self.client = None
            return

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set. Summarization will not be available.")
            self.client = None
            return

        try:
            # Create httpx client with proper SSL configuration
            http_client = httpx.Client(
                verify=certifi.where(),
                timeout=60.0
            )
            self.client = openai.OpenAI(
                api_key=api_key,
                http_client=http_client
            )
            logger.info(f"PaperSummarizer initialized with model: {self.model}")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI client: {e}")
            self.client = None

    def summarize_paper(
        self,
        text: str,
        max_length: int = 500,
        focus_areas: list = None
    ) -> Dict[str, str]:
        """
        Generate a structured summary of a scientific paper.

        Parameters
        ----------
        text : str
            Full text of the paper
        max_length : int
            Maximum summary length in words
        focus_areas : list
            Specific areas to focus on (e.g., ['methodology', 'results'])

        Returns
        -------
        Dict[str, str]
            Structured summary with sections
        """
        if not self.client:
            return self._generate_fallback_summary(text)

        # Truncate text if too long (GPT context limits)
        text_excerpt = self._extract_key_sections(text)

        focus_str = ""
        if focus_areas:
            focus_str = f"\nFocus particularly on: {', '.join(focus_areas)}"

        prompt = f"""Analyze this scientific paper and provide a structured summary.

Paper text:
{text_excerpt}

Please provide:
1. **Main Objective**: What is the primary goal of this research?
2. **Key Methods**: What approaches/techniques were used?
3. **Major Findings**: What are the main results and discoveries?
4. **Clinical Relevance**: What are the implications for drug development or treatment?
5. **Chemicals/Drugs Mentioned**: List key chemical compounds or drugs discussed.
{focus_str}

Keep the summary concise (max {max_length} words total) but informative."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a scientific research analyst specializing in pharmaceutical and biomedical research. Provide clear, accurate summaries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )

            summary_text = response.choices[0].message.content

            # Parse into structured format
            return self._parse_summary(summary_text)

        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return self._generate_fallback_summary(text)

    def _extract_key_sections(self, text: str, max_chars: int = 8000) -> str:
        """
        Extract key sections from paper text.

        Prioritizes abstract, introduction, methods, results, and conclusion.
        """
        # If text is short enough, return as-is
        if len(text) <= max_chars:
            return text

        # Try to find key sections
        sections = []
        section_keywords = [
            'abstract', 'introduction', 'background',
            'methods', 'methodology', 'materials',
            'results', 'findings',
            'discussion', 'conclusion'
        ]

        text_lower = text.lower()
        for keyword in section_keywords:
            # Look for section headers
            pattern = f'\n{keyword}'
            if pattern in text_lower:
                start = text_lower.find(pattern)
                # Extract up to 1000 chars from this section
                excerpt = text[start:start + 1000]
                sections.append(excerpt)

        if sections:
            combined = '\n\n'.join(sections)
            if len(combined) <= max_chars:
                return combined
            return combined[:max_chars]

        # Fallback: return first portion
        return text[:max_chars]

    def _parse_summary(self, summary_text: str) -> Dict[str, str]:
        """Parse structured summary from LLM output."""
        sections = {
            'objective': '',
            'methods': '',
            'findings': '',
            'relevance': '',
            'chemicals': '',
            'full_summary': summary_text
        }

        # Try to extract sections
        lines = summary_text.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for section headers
            if 'objective' in line.lower() and ':' in line:
                current_section = 'objective'
                line = line.split(':', 1)[1].strip() if ':' in line else ''
            elif 'method' in line.lower() and ':' in line:
                current_section = 'methods'
                line = line.split(':', 1)[1].strip() if ':' in line else ''
            elif 'finding' in line.lower() and ':' in line:
                current_section = 'findings'
                line = line.split(':', 1)[1].strip() if ':' in line else ''
            elif 'relevance' in line.lower() and ':' in line:
                current_section = 'relevance'
                line = line.split(':', 1)[1].strip() if ':' in line else ''
            elif 'chemical' in line.lower() and ':' in line:
                current_section = 'chemicals'
                line = line.split(':', 1)[1].strip() if ':' in line else ''

            # Append to current section
            if current_section and line:
                if sections[current_section]:
                    sections[current_section] += ' ' + line
                else:
                    sections[current_section] = line

        return sections

    def _generate_fallback_summary(self, text: str) -> Dict[str, str]:
        """Generate a basic summary when OpenAI is not available."""
        # Extract first few sentences as summary
        sentences = text.split('.')[:5]
        summary = '. '.join(sentences).strip() + '.'

        return {
            'objective': 'Summary not available (OpenAI API not configured)',
            'methods': '',
            'findings': '',
            'relevance': '',
            'chemicals': '',
            'full_summary': summary[:500] + '...'
        }
