import numpy as np
from sqlalchemy.orm import Session
from datetime import datetime

def update_user_type_weights(session: Session, user_id: str, type_id: int, interaction_coefficient: float) -> float:
    from models import UserPreferredType
    result = session.query(UserPreferredType).filter_by(user_id=user_id, type_id=type_id).first()
    current_weight = result.weight if result else 1.0
    new_weight = current_weight + interaction_coefficient
    new_weight = float(np.clip(new_weight, 0.3, 2.0))
    if result:
        result.weight = new_weight
        result.updated_at = datetime.utcnow()
    else:
        session.add(UserPreferredType(user_id=user_id, type_id=type_id, weight=new_weight, updated_at=datetime.utcnow()))
    session.commit()
    return new_weight