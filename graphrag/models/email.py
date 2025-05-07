from typing import Dict, List
from datetime import datetime
import json
import os

class Email:
    def __init__(
        self,
        id: str,
        from_addr: str,
        to_addr: str,
        subject: str,
        message: str,
        date: str
    ):
        self.id = id
        self.from_addr = from_addr
        self.to_addr = to_addr
        self.subject = subject
        self.message = message
        self.date = date
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'from': self.from_addr,
            'to': self.to_addr,
            'subject': self.subject,
            'message': self.message,
            'date': self.date
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Email':
        return cls(
            id=data['id'],
            from_addr=data['from'],
            to_addr=data['to'],
            subject=data['subject'],
            message=data['message'],
            date=data['date']
        )

class EmailCollection:
    def __init__(self, file_path: str = "emails.json"):
        self.file_path = file_path
        self.emails: List[Email] = []
    
    def add_email(self, email: Email):
        self.emails.append(email)
    
    def save(self):
        data = [email.to_dict() for email in self.emails]
        with open(self.file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as f:
                data = json.load(f)
                self.emails = [Email.from_dict(item) for item in data]
    
    def get_emails(self) -> List[Dict]:
        return [email.to_dict() for email in self.emails]
    
