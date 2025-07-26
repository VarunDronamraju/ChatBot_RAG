"""
User Service for RAGBot
Location: app/services/user_service.py

Handles user profile management, usage statistics, activity tracking, and preferences.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from uuid import UUID
from sqlalchemy import func, desc
from collections import Counter
import re

from ..rag_engine.db.models import (
    User, Conversation, Message, DocumentMetadata, 
    UsageStat, QueryLog, MessageFeedback, AuditLog
)
from ..utils.logger import get_logger

logger = get_logger()

class UserService:
    def __init__(self, db):
        self.db = db

    def get_user_profile(self, user_id: UUID) -> Optional[Dict[str, Any]]:
        try:
            user = self.db.query(User).filter(User.id == user_id).first()
            if not user:
                return None

            stats = self._aggregate_user_stats(user_id)
            return {
                "id": str(user.id),
                "email": user.email,
                "name": user.name,
                "picture_url": user.picture_url,
                "created_at": user.created_at,
                "last_login": user.last_login,
                "global_tags": user.global_tags or [],
                "search_bias_mode": user.search_bias_mode,
                "conversation_style": user.conversation_style or {},
                "usage_metrics": user.usage_metrics or {},
                "stats": stats
            }
        except Exception as e:
            logger.error(f"Get user profile error: {str(e)}")
            return None

    def _aggregate_user_stats(self, user_id: UUID) -> Dict[str, Any]:
        try:
            total_conversations = self.db.query(Conversation).filter(
                Conversation.user_id == user_id,
                Conversation.is_deleted == False
            ).count()

            total_messages = self.db.query(Message).join(Conversation).filter(
                Conversation.user_id == user_id,
                Message.role == "user"
            ).count()

            total_documents = self.db.query(DocumentMetadata).filter(
                DocumentMetadata.owner_user_id == user_id
            ).count()

            avg_response_time = self.db.query(func.avg(Message.response_time)).join(Conversation).filter(
                Conversation.user_id == user_id,
                Message.role == "assistant",
                Message.response_time.isnot(None)
            ).scalar() or 0.0

            return {
                "total_conversations": total_conversations,
                "total_messages": total_messages,
                "total_documents": total_documents,
                "avg_response_time": round(avg_response_time, 2)
            }
        except Exception as e:
            logger.error(f"Aggregate stats error: {str(e)}")
            return {}

    def get_usage_statistics(self, user_id: UUID) -> Dict[str, Any]:
        try:
            today = datetime.utcnow().date()
            return {
                "today": self._get_usage_stats_for_period(user_id, today, today),
                "week": self._get_usage_stats_for_period(user_id, today - timedelta(days=7), today),
                "month": self._get_usage_stats_for_period(user_id, today - timedelta(days=30), today),
                "total": self._get_usage_stats_for_period(user_id, None, None)
            }
        except Exception as e:
            logger.error(f"Get usage statistics error: {str(e)}")
            return {}

    def _get_usage_stats_for_period(self, user_id: UUID, start_date, end_date) -> Dict[str, Any]:
        try:
            conv_query = self.db.query(Conversation).filter(Conversation.user_id == user_id, Conversation.is_deleted == False)
            msg_query = self.db.query(Message).join(Conversation).filter(Conversation.user_id == user_id)

            if start_date and end_date:
                conv_query = conv_query.filter(func.date(Conversation.created_at).between(start_date, end_date))
                msg_query = msg_query.filter(func.date(Message.timestamp).between(start_date, end_date))

            user_msgs = msg_query.filter(Message.role == "user").count()
            assistant_msgs = msg_query.filter(Message.role == "assistant").count()
            avg_response = msg_query.filter(Message.role == "assistant", Message.response_time.isnot(None)).with_entities(func.avg(Message.response_time)).scalar() or 0.0
            token_sum = msg_query.filter(Message.token_count.isnot(None)).with_entities(func.sum(Message.token_count)).scalar() or 0

            return {
                "conversations": conv_query.count(),
                "user_messages": user_msgs,
                "assistant_messages": assistant_msgs,
                "total_messages": user_msgs + assistant_msgs,
                "avg_response_time": round(avg_response, 2),
                "total_tokens": token_sum
            }
        except Exception as e:
            logger.error(f"Get period stats error: {str(e)}")
            return {}

    def get_user_activity(self, user_id: UUID) -> Dict[str, Any]:
        try:
            stats = self._aggregate_user_stats(user_id)

            feature_usage = self.db.query(QueryLog.source, func.count(QueryLog.source)).filter(
                QueryLog.user_id == user_id).group_by(QueryLog.source).order_by(desc(func.count(QueryLog.source))).limit(5).all()
            most_used_features = [f"{src} ({count} uses)" for src, count in feature_usage]

            recent_conversations = self.db.query(Conversation).filter(
                Conversation.user_id == user_id, Conversation.is_deleted == False
            ).order_by(desc(Conversation.updated_at)).limit(10).all()

            activity = [{
                "type": "conversation",
                "title": c.title,
                "date": c.updated_at,
                "message_count": c.message_count,
                "tags": c.tags or []
            } for c in recent_conversations]

            recent_docs = self.db.query(DocumentMetadata).filter(
                DocumentMetadata.owner_user_id == user_id
            ).order_by(desc(DocumentMetadata.id)).limit(5).all()

            activity += [{
                "type": "document_upload",
                "filename": d.filename,
                "date": datetime.utcnow(),
                "tags": d.tags or []
            } for d in recent_docs]

            activity.sort(key=lambda x: x["date"], reverse=True)
            return {
                **stats,
                "most_used_features": most_used_features,
                "recent_activity": activity[:10]
            }
        except Exception as e:
            logger.error(f"Get user activity error: {str(e)}")
            return {}

    def update_usage_metrics(self, user_id: UUID, metrics: Dict[str, Any]):
        try:
            user = self.db.query(User).filter(User.id == user_id).first()
            if user:
                current_metrics = user.usage_metrics or {}
                current_metrics.update(metrics)
                user.usage_metrics = current_metrics
                self.db.commit()
        except Exception as e:
            logger.error(f"Update usage metrics error: {str(e)}")

    def record_daily_usage(self, user_id: UUID, token_usage: int = 0, message_count: int = 0, cost: float = 0.0):
        try:
            today = datetime.utcnow().date()
            usage_stat = self.db.query(UsageStat).filter(
                UsageStat.user_id == user_id,
                func.date(UsageStat.date) == today
            ).first()

            if not usage_stat:
                usage_stat = UsageStat(user_id=user_id, date=datetime.utcnow(), token_usage=0, message_count=0, cost=0.0)
                self.db.add(usage_stat)

            usage_stat.token_usage += token_usage
            usage_stat.message_count += message_count
            usage_stat.cost += cost
            self.db.commit()
        except Exception as e:
            logger.error(f"Record daily usage error: {str(e)}")

    def get_user_feedback_summary(self, user_id: UUID) -> Dict[str, Any]:
        try:
            feedback_stats = self.db.query(
                MessageFeedback.rating, func.count(MessageFeedback.rating)
            ).join(Message).join(Conversation).filter(
                Conversation.user_id == user_id
            ).group_by(MessageFeedback.rating).all()

            total = sum(c for _, c in feedback_stats)
            if total == 0:
                return {"total_feedback": 0, "average_rating": 0.0, "rating_distribution": {}}

            rating_sum = sum(r * c for r, c in feedback_stats)
            avg_rating = rating_sum / total
            return {
                "total_feedback": total,
                "average_rating": round(avg_rating, 2),
                "rating_distribution": {str(r): c for r, c in feedback_stats}
            }
        except Exception as e:
            logger.error(f"Get feedback summary error: {str(e)}")
            return {}

    def get_conversation_insights(self, user_id: UUID) -> Dict[str, Any]:
        try:
            hours = self.db.query(
                func.extract('hour', Message.timestamp).label('hour'),
                func.count(Message.id)
            ).join(Conversation).filter(
                Conversation.user_id == user_id,
                Message.role == "user"
            ).group_by(func.extract('hour', Message.timestamp)).order_by(desc(func.count(Message.id))).limit(3).all()

            tags = self.db.query(Conversation.tags).filter(
                Conversation.user_id == user_id, Conversation.tags.isnot(None)
            ).all()
            tag_list = [tag for sublist, in tags if sublist for tag in sublist]
            tag_counts = Counter(tag_list)

            avg_len = self.db.query(func.avg(Conversation.message_count)).filter(
                Conversation.user_id == user_id, Conversation.is_deleted == False
            ).scalar() or 0.0

            chat_types = self.db.query(
                Conversation.chat_type, func.count(Conversation.chat_type)
            ).filter(
                Conversation.user_id == user_id, Conversation.is_deleted == False
            ).group_by(Conversation.chat_type).order_by(desc(func.count(Conversation.chat_type))).limit(3).all()

            return {
                "most_active_hours": [f"{int(h)}:00" for h, _ in hours],
                "most_used_tags": [t for t, _ in tag_counts.most_common(5)],
                "avg_conversation_length": round(avg_len, 1),
                "preferred_chat_types": [ct for ct, _ in chat_types],
                "total_unique_tags": len(set(tag_list))
            }
        except Exception as e:
            logger.error(f"Get conversation insights error: {str(e)}")
            return {}

    def update_user_last_login(self, user_id: UUID):
        try:
            user = self.db.query(User).filter(User.id == user_id).first()
            if user:
                user.last_login = datetime.utcnow()
                self.db.commit()
        except Exception as e:
            logger.error(f"Update last login error: {str(e)}")

    def get_user_search_patterns(self, user_id: UUID) -> Dict[str, Any]:
        try:
            queries = self.db.query(QueryLog.question).filter(
                QueryLog.user_id == user_id
            ).order_by(desc(QueryLog.timestamp)).limit(50).all()

            words = []
            for q, in queries:
                words += [w for w in re.findall(r'\b\w+\b', q.lower())
                          if w not in {"the", "is", "are", "and", "how", "for", "with", "from", "what", "where", "when"} and len(w) > 2]

            top_keywords = [w for w, _ in Counter(words).most_common(10)]

            source_pref = self.db.query(QueryLog.source, func.count(QueryLog.source)).filter(
                QueryLog.user_id == user_id
            ).group_by(QueryLog.source).order_by(desc(func.count(QueryLog.source))).all()

            avg_latency = self.db.query(func.avg(QueryLog.latency_ms)).filter(
                QueryLog.user_id == user_id, QueryLog.latency_ms.isnot(None)
            ).scalar() or 0.0

            return {
                "top_keywords": top_keywords,
                "source_preferences": {s: c for s, c in source_pref},
                "avg_search_latency_ms": round(avg_latency, 2),
                "total_searches": len(queries)
            }
        except Exception as e:
            logger.error(f"Get search patterns error: {str(e)}")
            return {}
