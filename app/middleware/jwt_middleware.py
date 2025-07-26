from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from fastapi.requests import Request
import jwt
from jwt import ExpiredSignatureError, InvalidTokenError
from fastapi import HTTPException
import os
import sqlite3
from dotenv import load_dotenv

load_dotenv()
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "fallback-secret-key")

class JWTAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.secret = JWT_SECRET_KEY

    async def dispatch(self, request: Request, call_next):
        # Allow these paths without authentication
        excluded_paths = [
            "/api/v1/auth",
            "/docs", 
            "/redoc",
            "/openapi.json", 
            "/",
            "/api/info",
            "/api/v1/health"  # Add health endpoint
        ]
        
        # Check if path should be excluded
        for excluded_path in excluded_paths:
            if request.url.path.startswith(excluded_path):
                return await call_next(request)

        # Get authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401, 
                content={"detail": "Missing or invalid token"}
            )

        token = auth_header.split(" ")[1]

        try:
            # Decode JWT token
            payload = jwt.decode(token, self.secret, algorithms=["HS256"])
            
            # Verify token exists in database and is valid
            conn = sqlite3.connect("app/local/auth_cache.db")
            cursor = conn.execute(
                "SELECT is_valid FROM token_cache WHERE token = ?", 
                (token,)
            )
            result = cursor.fetchone()
            conn.close()
            
            if not result or not result[0]:
                return JSONResponse(
                    status_code=401, 
                    content={"detail": "Token has been invalidated"}
                )
            
            # Add user info to request state
            request.state.user_id = payload["user_id"]
            request.state.user_email = payload["email"]
            request.state.user_role = payload.get("role", "user")


            
        except ExpiredSignatureError:
            return JSONResponse(
                status_code=401, 
                content={"detail": "Token expired"}
            )
        except InvalidTokenError:
            return JSONResponse(
                status_code=401, 
                content={"detail": "Invalid token"}
            )
        except Exception as e:
            return JSONResponse(
                status_code=401, 
                content={"detail": "Token validation failed"}
            )

        return await call_next(request)