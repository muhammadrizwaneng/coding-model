import argparse
import json
from pathlib import Path

DATASET_DIR = Path("datasets")
SEED_FILE = DATASET_DIR / "seed.jsonl"
MAIN_DATASET = DATASET_DIR / "coding_dataset.jsonl"

SEED_EXAMPLES = [
    {
        "instruction": "Create a FastAPI endpoint for user login that returns a JWT token.",
        "input": "Validate email and password. Read JWT_SECRET from an environment variable.",
        "output": (
            "import os\n"
            "from datetime import datetime, timedelta, timezone\n"
            "from fastapi import FastAPI, HTTPException, status\n"
            "from jose import jwt\n"
            "from pydantic import BaseModel, EmailStr, Field\n\n"
            "app = FastAPI()\n"
            "JWT_SECRET = os.environ['JWT_SECRET']\n"
            "ALGORITHM = 'HS256'\n\n"
            "class LoginRequest(BaseModel):\n"
            "    email: EmailStr\n"
            "    password: str = Field(min_length=8)\n\n"
            "# Demo-only lookup. Replace with a real user store + password hash check.\n"
            "DEMO_USERS = {'admin@example.com': 'replace-me-with-hashed-password'}\n\n"
            "@app.post('/login')\n"
            "def login(data: LoginRequest):\n"
            "    stored = DEMO_USERS.get(data.email)\n"
            "    if stored is None or data.password != stored:\n"
            "        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid credentials')\n"
            "    expires = datetime.now(timezone.utc) + timedelta(hours=1)\n"
            "    token = jwt.encode({'sub': data.email, 'exp': expires}, JWT_SECRET, algorithm=ALGORITHM)\n"
            "    return {'access_token': token, 'token_type': 'bearer'}"
        ),
    },
    {
        "instruction": "Create a React Native button component with custom style properties.",
        "input": "Button should accept a title and an onPress function.",
        "output": (
            "import React from 'react';\n"
            "import { TouchableOpacity, Text, StyleSheet, ViewStyle, TextStyle } from 'react-native';\n\n"
            "type CustomButtonProps = {\n"
            "  title: string;\n"
            "  onPress: () => void;\n"
            "  style?: ViewStyle;\n"
            "  textStyle?: TextStyle;\n"
            "};\n\n"
            "export const CustomButton = ({ title, onPress, style, textStyle }: CustomButtonProps) => {\n"
            "  return (\n"
            "    <TouchableOpacity style={[styles.button, style]} onPress={onPress}>\n"
            "      <Text style={[styles.text, textStyle]}>{title}</Text>\n"
            "    </TouchableOpacity>\n"
            "  );\n"
            "};\n\n"
            "const styles = StyleSheet.create({\n"
            "  button: { backgroundColor: '#007AFF', padding: 12, borderRadius: 8, alignItems: 'center' },\n"
            "  text: { color: '#FFFFFF', fontSize: 16, fontWeight: 'bold' },\n"
            "});"
        ),
    },
    {
        "instruction": "Build a responsive React Native Card component for displaying product details.",
        "input": "Props: title, price, imageUrl, and description.",
        "output": (
            "import React from 'react';\n"
            "import { View, Text, Image, StyleSheet } from 'react-native';\n\n"
            "type ProductCardProps = {\n"
            "  title: string;\n"
            "  price: number;\n"
            "  imageUrl: string;\n"
            "  description: string;\n"
            "};\n\n"
            "export const ProductCard = ({ title, price, imageUrl, description }: ProductCardProps) => {\n"
            "  return (\n"
            "    <View style={styles.card}>\n"
            "      <Image source={{ uri: imageUrl }} style={styles.image} />\n"
            "      <Text style={styles.title}>{title}</Text>\n"
            "      <Text style={styles.price}>${price.toFixed(2)}</Text>\n"
            "      <Text style={styles.desc}>{description}</Text>\n"
            "    </View>\n"
            "  );\n"
            "};\n\n"
            "const styles = StyleSheet.create({\n"
            "  card: { padding: 16, borderRadius: 10, backgroundColor: '#fff', marginVertical: 8 },\n"
            "  image: { width: '100%', height: 150, borderRadius: 8 },\n"
            "  title: { fontSize: 18, fontWeight: 'bold', marginTop: 8 },\n"
            "  price: { fontSize: 16, color: '#0a7', marginVertical: 4 },\n"
            "  desc: { fontSize: 14, color: '#666' },\n"
            "});"
        ),
    },
    {
        "instruction": "Write a PostgreSQL database connection dependency utility using SQLAlchemy in FastAPI.",
        "input": "Include a yielding session generator. Read DATABASE_URL from the environment.",
        "output": (
            "import os\n"
            "from sqlalchemy import create_engine\n"
            "from sqlalchemy.orm import sessionmaker, declarative_base\n\n"
            "DATABASE_URL = os.environ['DATABASE_URL']\n"
            "engine = create_engine(DATABASE_URL)\n"
            "SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)\n"
            "Base = declarative_base()\n\n"
            "def get_db():\n"
            "    db = SessionLocal()\n"
            "    try:\n"
            "        yield db\n"
            "    finally:\n"
            "        db.close()"
        ),
    },
    {
        "instruction": "Create FastAPI CORS middleware configured for a known frontend origin.",
        "input": "Allow https://app.example.com only. Do not use allow_origins=['*'] with credentials.",
        "output": (
            "from fastapi import FastAPI\n"
            "from fastapi.middleware.cors import CORSMiddleware\n\n"
            "app = FastAPI()\n"
            "app.add_middleware(\n"
            "    CORSMiddleware,\n"
            "    allow_origins=['https://app.example.com'],\n"
            "    allow_credentials=True,\n"
            "    allow_methods=['GET', 'POST', 'PUT', 'PATCH', 'DELETE'],\n"
            "    allow_headers=['Authorization', 'Content-Type'],\n"
            ")"
        ),
    },
]


def write_jsonl(records: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for item in records:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write seed training examples without overwriting the main dataset."
    )
    parser.add_argument(
        "--output",
        default=SEED_FILE,
        type=Path,
        help=f"Where to write seed examples (default: {SEED_FILE}).",
    )
    parser.add_argument(
        "--force-main",
        action="store_true",
        help=(
            f"DANGEROUS: overwrite {MAIN_DATASET} with only the seed examples. "
            "Refused unless you also pass --i-understand-this-deletes-data."
        ),
    )
    parser.add_argument(
        "--i-understand-this-deletes-data",
        action="store_true",
        help="Required confirmation when using --force-main.",
    )
    args = parser.parse_args()

    if args.force_main:
        if not args.i_understand_this_deletes_data:
            print(
                "Refusing to overwrite the main dataset. "
                "Pass --force-main --i-understand-this-deletes-data to confirm, "
                f"or write seeds to {SEED_FILE} (default)."
            )
            return 1
        write_jsonl(SEED_EXAMPLES, MAIN_DATASET)
        print(f"Overwrote {MAIN_DATASET} with {len(SEED_EXAMPLES)} seed records.")
        return 0

    write_jsonl(SEED_EXAMPLES, args.output)
    print(f"Wrote {len(SEED_EXAMPLES)} seed records to {args.output}")
    print(f"Main dataset left untouched: {MAIN_DATASET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
