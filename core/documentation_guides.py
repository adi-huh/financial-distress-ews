"""
Day 25: Documentation & Guides Generation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class DocumentSection:
    """Documentation section"""
    title: str
    content: str
    subsections: List['DocumentSection'] = field(default_factory=list)
    level: int = 1


class DocumentationGenerator:
    """Generate documentation"""
    
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.sections = []
        self.metadata = {}
    
    def add_section(self, section: DocumentSection):
        """Add documentation section"""
        self.sections.append(section)
    
    def set_metadata(self, key: str, value: str):
        """Set metadata"""
        self.metadata[key] = value
    
    def generate_markdown(self) -> str:
        """Generate Markdown documentation"""
        content = f"# {self.project_name}\n\n"
        for section in self.sections:
            content += self._render_section(section)
        return content
    
    def _render_section(self, section: DocumentSection) -> str:
        """Render a section"""
        heading = "#" * section.level + " " + section.title + "\n\n"
        content = heading + section.content + "\n\n"
        for subsection in section.subsections:
            content += self._render_section(subsection)
        return content
    
    def generate_html(self) -> str:
        """Generate HTML documentation"""
        html = f"<html><head><title>{self.project_name}</title></head><body>\n"
        for section in self.sections:
            html += self._render_html_section(section)
        html += "</body></html>"
        return html
    
    def _render_html_section(self, section: DocumentSection) -> str:
        """Render HTML section"""
        tag = f"h{section.level}"
        html = f"<{tag}>{section.title}</{tag}>\n"
        html += f"<p>{section.content}</p>\n"
        for subsection in section.subsections:
            html += self._render_html_section(subsection)
        return html


class APIGuideGenerator:
    """Generate API guide"""
    
    def __init__(self):
        self.endpoints = []
        self.examples = []
    
    def add_endpoint(self, method: str, path: str, description: str, parameters: Dict[str, str]):
        """Add endpoint documentation"""
        self.endpoints.append({
            "method": method,
            "path": path,
            "description": description,
            "parameters": parameters
        })
    
    def add_example(self, title: str, request: str, response: str):
        """Add usage example"""
        self.examples.append({
            "title": title,
            "request": request,
            "response": response
        })
    
    def generate_guide(self) -> str:
        """Generate API guide"""
        guide = "# API Guide\n\n"
        guide += "## Endpoints\n\n"
        for endpoint in self.endpoints:
            guide += f"### {endpoint['method']} {endpoint['path']}\n"
            guide += f"{endpoint['description']}\n\n"
        
        guide += "## Examples\n\n"
        for example in self.examples:
            guide += f"### {example['title']}\n"
            guide += f"Request: {example['request']}\n"
            guide += f"Response: {example['response']}\n\n"
        
        return guide


class UserGuideGenerator:
    """Generate user guide"""
    
    def __init__(self):
        self.chapters = []
    
    def add_chapter(self, title: str, content: str):
        """Add chapter"""
        self.chapters.append({"title": title, "content": content})
    
    def generate_guide(self) -> str:
        """Generate user guide"""
        guide = "# User Guide\n\n"
        guide += "## Table of Contents\n\n"
        for i, chapter in enumerate(self.chapters, 1):
            guide += f"{i}. {chapter['title']}\n"
        
        guide += "\n## Chapters\n\n"
        for chapter in self.chapters:
            guide += f"## {chapter['title']}\n"
            guide += f"{chapter['content']}\n\n"
        
        return guide


class DeploymentGuide:
    """Deployment guide"""
    
    def __init__(self):
        self.prerequisites = []
        self.steps = []
        self.troubleshooting = {}
    
    def add_prerequisite(self, name: str, description: str):
        """Add prerequisite"""
        self.prerequisites.append({"name": name, "description": description})
    
    def add_step(self, step_num: int, title: str, instructions: str):
        """Add deployment step"""
        self.steps.append({
            "number": step_num,
            "title": title,
            "instructions": instructions
        })
    
    def add_troubleshooting(self, issue: str, solution: str):
        """Add troubleshooting entry"""
        self.troubleshooting[issue] = solution
    
    def generate_guide(self) -> str:
        """Generate deployment guide"""
        guide = "# Deployment Guide\n\n"
        
        guide += "## Prerequisites\n\n"
        for prereq in self.prerequisites:
            guide += f"- {prereq['name']}: {prereq['description']}\n"
        
        guide += "\n## Deployment Steps\n\n"
        for step in self.steps:
            guide += f"### Step {step['number']}: {step['title']}\n"
            guide += f"{step['instructions']}\n\n"
        
        guide += "## Troubleshooting\n\n"
        for issue, solution in self.troubleshooting.items():
            guide += f"### {issue}\n"
            guide += f"{solution}\n\n"
        
        return guide


class ChangelogGenerator:
    """Generate changelog"""
    
    def __init__(self):
        self.versions = []
    
    def add_version(self, version: str, date: str, changes: List[str]):
        """Add version entry"""
        self.versions.append({
            "version": version,
            "date": date,
            "changes": changes
        })
    
    def generate_changelog(self) -> str:
        """Generate changelog"""
        changelog = "# Changelog\n\n"
        
        for entry in self.versions:
            changelog += f"## [{entry['version']}] - {entry['date']}\n\n"
            for change in entry['changes']:
                changelog += f"- {change}\n"
            changelog += "\n"
        
        return changelog


class TutorialGenerator:
    """Generate tutorials"""
    
    def __init__(self):
        self.tutorials = []
    
    def add_tutorial(self, title: str, level: str, sections: List[Dict[str, str]]):
        """Add tutorial"""
        self.tutorials.append({
            "title": title,
            "level": level,
            "sections": sections
        })
    
    def generate_tutorial(self, title: str) -> str:
        """Generate specific tutorial"""
        for tutorial in self.tutorials:
            if tutorial['title'] == title:
                content = f"# {tutorial['title']}\n"
                content += f"Level: {tutorial['level']}\n\n"
                for section in tutorial['sections']:
                    content += f"## {section['title']}\n"
                    content += f"{section['content']}\n\n"
                return content
        return ""
    
    def generate_all_tutorials(self) -> str:
        """Generate all tutorials"""
        content = "# Tutorials\n\n"
        for tutorial in self.tutorials:
            content += self.generate_tutorial(tutorial['title'])
        return content


class DocumentationPortal:
    """Documentation portal"""
    
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.docs = {}
        self.created_at = datetime.now().isoformat()
    
    def register_documentation(self, name: str, content: str):
        """Register documentation"""
        self.docs[name] = content
    
    def get_documentation(self, name: str) -> Optional[str]:
        """Get documentation"""
        return self.docs.get(name)
    
    def list_documentation(self) -> List[str]:
        """List all documentation"""
        return list(self.docs.keys())
    
    def generate_portal_index(self) -> str:
        """Generate portal index"""
        index = f"# {self.project_name} Documentation\n\n"
        index += "## Available Documentation\n\n"
        for doc_name in self.list_documentation():
            index += f"- [{doc_name}](#{doc_name.lower().replace(' ', '-')})\n"
        return index
