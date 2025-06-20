import os
from sqlalchemy.ext.asyncio import create_async_engine , async_sessionmaker
from  dotenv import load_dotenv


load_dotenv()

POSTGRES_URL = os.getenv("SQL_POSTGRES_URL")

engine = create_async_engine(POSTGRES_URL)

AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)