import asyncio

from sqlalchemy import text

from models.db import AsyncSessionLocal
from models.db import init_db


async def main():

    await init_db()

    async with AsyncSessionLocal() as db:

        result = await db.execute(
            text("SELECT 1")
        )

        print(
            result.scalar()
        )

        print(
            "\nDatabase connection successful"
        )

        print(
            "\nTables initialized"
        )


if __name__ == "__main__":

    asyncio.run(main())