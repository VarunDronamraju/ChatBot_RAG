from fastapi import Depends, HTTPException, Request

def require_role(allowed_roles: list):
    def role_checker(request: Request):
        user = getattr(request.state, "user", None)
        if not user:
            raise HTTPException(status_code=401, detail="User not authenticated")
        if user.get("role") not in allowed_roles:
            raise HTTPException(status_code=403, detail="Permission denied")
        return user
    return role_checker
