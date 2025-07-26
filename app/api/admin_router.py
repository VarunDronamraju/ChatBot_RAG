from fastapi import APIRouter, Depends
from app.dependencies.roles import require_role

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])

@router.get("/dashboard", dependencies=[Depends(require_role(["admin"]))])
def get_admin_data():
    return {"msg": "Only admin can access this."}
