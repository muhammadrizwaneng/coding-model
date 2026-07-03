import json
from pathlib import Path

RAW_DIR = Path("datasets/raw")

BATCHES = {
    "python_fastapi.jsonl": [
        {
            "instruction": "Create a FastAPI health check endpoint.",
            "input": "",
            "output": "from fastapi import FastAPI\n\napp = FastAPI()\n\n@app.get('/health')\ndef health():\n    return {'status': 'ok'}",
        },
        {
            "instruction": "Create a FastAPI user registration endpoint with Pydantic validation.",
            "input": "Fields: email, password, full_name.",
            "output": "from fastapi import FastAPI, HTTPException\nfrom pydantic import BaseModel, EmailStr, Field\n\napp = FastAPI()\n\nclass UserCreate(BaseModel):\n    email: EmailStr\n    password: str = Field(min_length=8)\n    full_name: str = Field(min_length=2)\n\nusers = []\n\n@app.post('/register', status_code=201)\ndef register(user: UserCreate):\n    if any(existing['email'] == user.email for existing in users):\n        raise HTTPException(status_code=400, detail='Email already exists')\n    users.append(user.model_dump())\n    return {'message': 'User registered'}",
        },
        {
            "instruction": "Create a FastAPI CRUD API for tasks using an in-memory list.",
            "input": "Task fields: id, title, completed.",
            "output": "from fastapi import FastAPI, HTTPException\nfrom pydantic import BaseModel\n\napp = FastAPI()\n\nclass Task(BaseModel):\n    id: int\n    title: str\n    completed: bool = False\n\ntasks: list[Task] = []\n\n@app.post('/tasks', status_code=201)\ndef create_task(task: Task):\n    tasks.append(task)\n    return task\n\n@app.get('/tasks')\ndef list_tasks():\n    return tasks\n\n@app.get('/tasks/{task_id}')\ndef get_task(task_id: int):\n    for task in tasks:\n        if task.id == task_id:\n            return task\n    raise HTTPException(status_code=404, detail='Task not found')",
        },
        {
            "instruction": "Create a FastAPI endpoint with pagination query parameters.",
            "input": "",
            "output": "from fastapi import FastAPI, Query\n\napp = FastAPI()\nitems = [f'item-{i}' for i in range(1, 51)]\n\n@app.get('/items')\ndef list_items(page: int = Query(1, ge=1), size: int = Query(10, ge=1, le=20)):\n    start = (page - 1) * size\n    end = start + size\n    return {'page': page, 'size': size, 'items': items[start:end]}",
        },
        {
            "instruction": "Create a FastAPI file upload endpoint.",
            "input": "Return filename and file size.",
            "output": "from fastapi import FastAPI, UploadFile, File\n\napp = FastAPI()\n\n@app.post('/upload')\nasync def upload_file(file: UploadFile = File(...)):\n    content = await file.read()\n    return {'filename': file.filename, 'size': len(content)}",
        },
        {
            "instruction": "Create a FastAPI login endpoint that returns a mock JWT token.",
            "input": "",
            "output": "from fastapi import FastAPI, HTTPException\nfrom pydantic import BaseModel\n\napp = FastAPI()\n\nclass LoginRequest(BaseModel):\n    email: str\n    password: str\n\n@app.post('/login')\ndef login(data: LoginRequest):\n    if data.email == 'admin@example.com' and data.password == 'secret123':\n        return {'access_token': 'mock-jwt-token', 'token_type': 'bearer'}\n    raise HTTPException(status_code=401, detail='Invalid credentials')",
        },
        {
            "instruction": "Create a FastAPI dependency for a SQLAlchemy database session.",
            "input": "Use PostgreSQL connection string.",
            "output": "from sqlalchemy import create_engine\nfrom sqlalchemy.orm import sessionmaker, declarative_base\n\nDATABASE_URL = 'postgresql://user:password@localhost:5432/appdb'\nengine = create_engine(DATABASE_URL)\nSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)\nBase = declarative_base()\n\ndef get_db():\n    db = SessionLocal()\n    try:\n        yield db\n    finally:\n        db.close()",
        },
        {
            "instruction": "Create a FastAPI custom exception handler for not found errors.",
            "input": "",
            "output": "from fastapi import FastAPI, Request\nfrom fastapi.responses import JSONResponse\n\napp = FastAPI()\n\nclass ItemNotFound(Exception):\n    pass\n\n@app.exception_handler(ItemNotFound)\nasync def item_not_found_handler(request: Request, exc: ItemNotFound):\n    return JSONResponse(status_code=404, content={'error': 'Item not found'})",
        },
        {
            "instruction": "Create a FastAPI endpoint that filters products by category query parameter.",
            "input": "",
            "output": "from fastapi import FastAPI, Query\n\napp = FastAPI()\nproducts = [\n    {'id': 1, 'name': 'Laptop', 'category': 'electronics'},\n    {'id': 2, 'name': 'Desk', 'category': 'furniture'},\n]\n\n@app.get('/products')\ndef list_products(category: str | None = Query(default=None)):\n    if category is None:\n        return products\n    return [product for product in products if product['category'] == category]",
        },
        {
            "instruction": "Create a FastAPI endpoint that triggers a background task.",
            "input": "",
            "output": "from fastapi import BackgroundTasks, FastAPI\n\napp = FastAPI()\n\ndef write_log(message: str) -> None:\n    print(f'LOG: {message}')\n\n@app.post('/notify')\ndef notify(message: str, background_tasks: BackgroundTasks):\n    background_tasks.add_task(write_log, message)\n    return {'status': 'queued'}",
        },
        {
            "instruction": "Create a FastAPI endpoint for bulk creating products.",
            "input": "Accept a list of products in the request body.",
            "output": "from fastapi import FastAPI\nfrom pydantic import BaseModel\n\napp = FastAPI()\n\nclass Product(BaseModel):\n    name: str\n    price: float\n\nproducts: list[Product] = []\n\n@app.post('/products/bulk', status_code=201)\ndef bulk_create(items: list[Product]):\n    products.extend(items)\n    return {'created': len(items), 'total': len(products)}",
        },
        {
            "instruction": "Create a FastAPI soft delete endpoint for users.",
            "input": "Mark user as inactive instead of deleting.",
            "output": "from fastapi import FastAPI, HTTPException\n\napp = FastAPI()\nusers = [{'id': 1, 'name': 'Alice', 'active': True}]\n\n@app.delete('/users/{user_id}')\ndef soft_delete_user(user_id: int):\n    for user in users:\n        if user['id'] == user_id:\n            user['active'] = False\n            return {'message': 'User deactivated'}\n    raise HTTPException(status_code=404, detail='User not found')",
        },
        {
            "instruction": "Create a FastAPI search endpoint for books by title.",
            "input": "",
            "output": "from fastapi import FastAPI, Query\n\napp = FastAPI()\nbooks = [{'id': 1, 'title': 'Clean Code'}, {'id': 2, 'title': 'FastAPI Guide'}]\n\n@app.get('/books/search')\ndef search_books(q: str = Query(min_length=1)):\n    results = [book for book in books if q.lower() in book['title'].lower()]\n    return results",
        },
        {
            "instruction": "Create a FastAPI app with CORS middleware enabled.",
            "input": "Allow all origins for development.",
            "output": "from fastapi import FastAPI\nfrom fastapi.middleware.cors import CORSMiddleware\n\napp = FastAPI()\napp.add_middleware(\n    CORSMiddleware,\n    allow_origins=['*'],\n    allow_credentials=True,\n    allow_methods=['*'],\n    allow_headers=['*'],\n)",
        },
        {
            "instruction": "Create a versioned FastAPI route under /api/v1.",
            "input": "",
            "output": "from fastapi import APIRouter, FastAPI\n\napp = FastAPI()\nrouter = APIRouter(prefix='/api/v1')\n\n@router.get('/status')\ndef status():\n    return {'version': 'v1', 'status': 'ok'}\n\napp.include_router(router)",
        },
        {
            "instruction": "Create a FastAPI middleware that adds a request ID header.",
            "input": "",
            "output": "import uuid\nfrom fastapi import FastAPI, Request\n\napp = FastAPI()\n\n@app.middleware('http')\nasync def add_request_id(request: Request, call_next):\n    request_id = str(uuid.uuid4())\n    response = await call_next(request)\n    response.headers['X-Request-ID'] = request_id\n    return response",
        },
        {
            "instruction": "Create a FastAPI endpoint that validates a webhook signature header.",
            "input": "",
            "output": "from fastapi import FastAPI, Header, HTTPException\n\napp = FastAPI()\nSECRET = 'webhook-secret'\n\n@app.post('/webhook')\ndef receive_webhook(payload: dict, x_signature: str = Header(...)):\n    if x_signature != SECRET:\n        raise HTTPException(status_code=401, detail='Invalid signature')\n    return {'received': True, 'payload': payload}",
        },
        {
            "instruction": "Create a FastAPI endpoint that returns 422 for invalid email format using Pydantic.",
            "input": "",
            "output": "from fastapi import FastAPI\nfrom pydantic import BaseModel, EmailStr\n\napp = FastAPI()\n\nclass Contact(BaseModel):\n    email: EmailStr\n    message: str\n\n@app.post('/contact')\ndef contact(data: Contact):\n    return {'message': 'Message received', 'email': data.email}",
        },
        {
            "instruction": "Create a FastAPI endpoint that updates only provided fields for a product.",
            "input": "Use partial update pattern.",
            "output": "from fastapi import FastAPI, HTTPException\nfrom pydantic import BaseModel\n\napp = FastAPI()\n\nclass ProductUpdate(BaseModel):\n    name: str | None = None\n    price: float | None = None\n\nproducts = {1: {'id': 1, 'name': 'Phone', 'price': 699.0}}\n\n@app.patch('/products/{product_id}')\ndef update_product(product_id: int, data: ProductUpdate):\n    product = products.get(product_id)\n    if not product:\n        raise HTTPException(status_code=404, detail='Product not found')\n    for key, value in data.model_dump(exclude_none=True).items():\n        product[key] = value\n    return product",
        },
        {
            "instruction": "Create a FastAPI SQLAlchemy Product model and CRUD create endpoint.",
            "input": "Fields: id, name, price.",
            "output": "from fastapi import Depends, FastAPI\nfrom pydantic import BaseModel\nfrom sqlalchemy import Column, Float, Integer, String\nfrom sqlalchemy.orm import Session, declarative_base\n\nBase = declarative_base()\n\nclass Product(Base):\n    __tablename__ = 'products'\n    id = Column(Integer, primary_key=True)\n    name = Column(String, nullable=False)\n    price = Column(Float, nullable=False)\n\nclass ProductCreate(BaseModel):\n    name: str\n    price: float\n\napp = FastAPI()\n\ndef get_db():\n    raise NotImplementedError('Provide SessionLocal dependency')\n\n@app.post('/products', status_code=201)\ndef create_product(data: ProductCreate, db: Session = Depends(get_db)):\n    product = Product(name=data.name, price=data.price)\n    db.add(product)\n    db.commit()\n    db.refresh(product)\n    return {'id': product.id, 'name': product.name, 'price': product.price}",
        },
    ],
    "node_express.jsonl": [
        {
            "instruction": "Create a basic Express.js server with a JSON health endpoint.",
            "input": "",
            "output": "const express = require('express');\nconst app = express();\napp.use(express.json());\n\napp.get('/health', (req, res) => {\n  res.status(200).json({ status: 'ok' });\n});\n\napp.listen(3000, () => console.log('Server running on port 3000'));",
        },
        {
            "instruction": "Create a Node.js Express CRUD API for appointments using MongoDB and Mongoose.",
            "input": "Include get by id endpoint and status enum validation.",
            "output": "const express = require('express');\nconst mongoose = require('mongoose');\n\nconst app = express();\napp.use(express.json());\n\nmongoose.connect('mongodb://localhost:27017/appointments');\n\nconst appointmentSchema = new mongoose.Schema({\n  patient_name: { type: String, required: true },\n  doctor_name: { type: String, required: true },\n  appointment_date: { type: Date, required: true },\n  status: { type: String, enum: ['scheduled', 'completed', 'cancelled'], required: true },\n});\n\nconst Appointment = mongoose.model('Appointment', appointmentSchema);\n\napp.get('/appointments', async (req, res) => {\n  const items = await Appointment.find();\n  res.json(items);\n});\n\napp.get('/appointments/:id', async (req, res) => {\n  const item = await Appointment.findById(req.params.id);\n  if (!item) return res.status(404).json({ message: 'Not found' });\n  res.json(item);\n});\n\napp.post('/appointments', async (req, res) => {\n  const item = await Appointment.create(req.body);\n  res.status(201).json(item);\n});\n\napp.listen(3000);",
        },
        {
            "instruction": "Create Express middleware that logs request method and URL.",
            "input": "",
            "output": "function requestLogger(req, res, next) {\n  console.log(`${req.method} ${req.url}`);\n  next();\n}\n\nmodule.exports = requestLogger;",
        },
        {
            "instruction": "Create an Express JWT login route.",
            "input": "Return token on valid credentials.",
            "output": "const express = require('express');\nconst jwt = require('jsonwebtoken');\n\nconst app = express();\napp.use(express.json());\nconst SECRET = 'dev-secret';\n\napp.post('/login', (req, res) => {\n  const { email, password } = req.body;\n  if (email === 'admin@example.com' && password === 'secret123') {\n    const token = jwt.sign({ email }, SECRET, { expiresIn: '1h' });\n    return res.json({ access_token: token });\n  }\n  return res.status(401).json({ message: 'Invalid credentials' });\n});",
        },
        {
            "instruction": "Create a global Express error handling middleware.",
            "input": "",
            "output": "function errorHandler(err, req, res, next) {\n  console.error(err);\n  res.status(err.status || 500).json({ message: err.message || 'Internal Server Error' });\n}\n\nmodule.exports = errorHandler;",
        },
        {
            "instruction": "Create Express route validation middleware for required body fields.",
            "input": "Required fields: name, email.",
            "output": "function validateUser(req, res, next) {\n  const { name, email } = req.body;\n  if (!name || !email) {\n    return res.status(400).json({ message: 'name and email are required' });\n  }\n  next();\n}\n\nmodule.exports = validateUser;",
        },
        {
            "instruction": "Create an Express API that queries PostgreSQL using pg Pool.",
            "input": "Return all users.",
            "output": "const express = require('express');\nconst { Pool } = require('pg');\n\nconst app = express();\nconst pool = new Pool({ connectionString: 'postgresql://user:password@localhost:5432/appdb' });\n\napp.get('/users', async (req, res) => {\n  const result = await pool.query('SELECT id, name, email FROM users');\n  res.json(result.rows);\n});",
        },
        {
            "instruction": "Create an Express 404 not found middleware.",
            "input": "",
            "output": "function notFound(req, res, next) {\n  res.status(404).json({ message: 'Route not found' });\n}\n\nmodule.exports = notFound;",
        },
        {
            "instruction": "Create an Express users CRUD API with in-memory storage.",
            "input": "",
            "output": "const express = require('express');\nconst app = express();\napp.use(express.json());\n\nlet users = [];\n\napp.post('/users', (req, res) => {\n  const user = { id: users.length + 1, ...req.body };\n  users.push(user);\n  res.status(201).json(user);\n});\n\napp.get('/users', (req, res) => res.json(users));\n\napp.get('/users/:id', (req, res) => {\n  const user = users.find((item) => item.id === Number(req.params.id));\n  if (!user) return res.status(404).json({ message: 'Not found' });\n  res.json(user);\n});",
        },
        {
            "instruction": "Create an Express CORS middleware setup for development.",
            "input": "",
            "output": "const cors = require('cors');\n\nconst corsOptions = {\n  origin: '*',\n  methods: ['GET', 'POST', 'PUT', 'PATCH', 'DELETE'],\n  allowedHeaders: ['Content-Type', 'Authorization'],\n};\n\nmodule.exports = cors(corsOptions);",
        },
        {
            "instruction": "Create an Express endpoint with pagination for products.",
            "input": "",
            "output": "const express = require('express');\nconst app = express();\nconst products = Array.from({ length: 30 }, (_, i) => ({ id: i + 1, name: `Product ${i + 1}` }));\n\napp.get('/products', (req, res) => {\n  const page = Number(req.query.page || 1);\n  const size = Number(req.query.size || 10);\n  const start = (page - 1) * size;\n  res.json({ page, size, items: products.slice(start, start + size) });\n});",
        },
        {
            "instruction": "Create an Express API key authentication middleware.",
            "input": "",
            "output": "function apiKeyAuth(req, res, next) {\n  const apiKey = req.header('x-api-key');\n  if (apiKey !== 'dev-api-key') {\n    return res.status(401).json({ message: 'Unauthorized' });\n  }\n  next();\n}\n\nmodule.exports = apiKeyAuth;",
        },
        {
            "instruction": "Create an Express route that handles async errors with try/catch.",
            "input": "",
            "output": "const express = require('express');\nconst app = express();\n\napp.get('/profile/:id', async (req, res) => {\n  try {\n  const profile = await Promise.resolve({ id: req.params.id, name: 'Alice' });\n  res.json(profile);\n  } catch (error) {\n  res.status(500).json({ message: error.message });\n  }\n});",
        },
        {
            "instruction": "Create an Express environment config loader using dotenv.",
            "input": "",
            "output": "require('dotenv').config();\n\nconst config = {\n  port: process.env.PORT || 3000,\n  databaseUrl: process.env.DATABASE_URL,\n  jwtSecret: process.env.JWT_SECRET,\n};\n\nmodule.exports = config;",
        },
        {
            "instruction": "Create an Express DELETE endpoint that returns 204 on success.",
            "input": "",
            "output": "const express = require('express');\nconst app = express();\nlet items = [{ id: 1, name: 'Item 1' }];\n\napp.delete('/items/:id', (req, res) => {\n  const index = items.findIndex((item) => item.id === Number(req.params.id));\n  if (index === -1) return res.status(404).json({ message: 'Not found' });\n  items.splice(index, 1);\n  return res.status(204).send();\n});",
        },
        {
            "instruction": "Create an Express search endpoint for customers by name.",
            "input": "",
            "output": "const express = require('express');\nconst app = express();\nconst customers = [{ id: 1, name: 'Alice' }, { id: 2, name: 'Bob' }];\n\napp.get('/customers/search', (req, res) => {\n  const q = String(req.query.q || '').toLowerCase();\n  const results = customers.filter((customer) => customer.name.toLowerCase().includes(q));\n  res.json(results);\n});",
        },
        {
            "instruction": "Create an Express route that validates status enum before update.",
            "input": "Allowed values: active, inactive.",
            "output": "const express = require('express');\nconst app = express();\napp.use(express.json());\n\napp.put('/users/:id/status', (req, res) => {\n  const allowed = ['active', 'inactive'];\n  const { status } = req.body;\n  if (!allowed.includes(status)) {\n    return res.status(400).json({ message: 'Invalid status' });\n  }\n  return res.json({ id: req.params.id, status });\n});",
        },
        {
            "instruction": "Create an Express router module for product routes.",
            "input": "",
            "output": "const express = require('express');\nconst router = express.Router();\n\nrouter.get('/', (req, res) => {\n  res.json([{ id: 1, name: 'Laptop' }]);\n});\n\nrouter.post('/', (req, res) => {\n  res.status(201).json({ id: 2, ...req.body });\n});\n\nmodule.exports = router;",
        },
        {
            "instruction": "Create an Express middleware that parses JSON request bodies.",
            "input": "",
            "output": "const express = require('express');\nconst app = express();\n\napp.use(express.json());\n\napp.post('/echo', (req, res) => {\n  res.json(req.body);\n});",
        },
        {
            "instruction": "Create an Express graceful shutdown handler.",
            "input": "",
            "output": "const server = app.listen(3000);\n\nprocess.on('SIGTERM', () => {\n  console.log('Shutting down gracefully');\n  server.close(() => process.exit(0));\n});",
        },
    ],
    "react.jsonl": [
        {
            "instruction": "Create a reusable React button component.",
            "input": "Props: label, onClick, disabled.",
            "output": "export function Button({ label, onClick, disabled = false }) {\n  return (\n    <button onClick={onClick} disabled={disabled}>\n      {label}\n    </button>\n  );\n}",
        },
        {
            "instruction": "Create a React login form with email and password fields.",
            "input": "",
            "output": "import { useState } from 'react';\n\nexport function LoginForm({ onSubmit }) {\n  const [email, setEmail] = useState('');\n  const [password, setPassword] = useState('');\n\n  return (\n    <form onSubmit={(event) => { event.preventDefault(); onSubmit({ email, password }); }}>\n      <input value={email} onChange={(e) => setEmail(e.target.value)} placeholder='Email' />\n      <input type='password' value={password} onChange={(e) => setPassword(e.target.value)} placeholder='Password' />\n      <button type='submit'>Login</button>\n    </form>\n  );\n}",
        },
        {
            "instruction": "Create a React product card component.",
            "input": "Show title, price, and image.",
            "output": "export function ProductCard({ title, price, imageUrl }) {\n  return (\n    <div className='product-card'>\n      <img src={imageUrl} alt={title} />\n      <h3>{title}</h3>\n      <p>${price}</p>\n    </div>\n  );\n}",
        },
        {
            "instruction": "Create a React counter component using useState.",
            "input": "",
            "output": "import { useState } from 'react';\n\nexport function Counter() {\n  const [count, setCount] = useState(0);\n  return (\n    <div>\n      <p>Count: {count}</p>\n      <button onClick={() => setCount(count + 1)}>Increment</button>\n    </div>\n  );\n}",
        },
        {
            "instruction": "Create a React component that fetches users on mount with useEffect.",
            "input": "",
            "output": "import { useEffect, useState } from 'react';\n\nexport function UserList() {\n  const [users, setUsers] = useState([]);\n\n  useEffect(() => {\n    fetch('/api/users')\n      .then((res) => res.json())\n      .then(setUsers);\n  }, []);\n\n  return <ul>{users.map((user) => <li key={user.id}>{user.name}</li>)}</ul>;\n}",
        },
        {
            "instruction": "Create a custom React useFetch hook.",
            "input": "",
            "output": "import { useEffect, useState } from 'react';\n\nexport function useFetch(url) {\n  const [data, setData] = useState(null);\n  const [loading, setLoading] = useState(true);\n  const [error, setError] = useState(null);\n\n  useEffect(() => {\n    fetch(url)\n      .then((res) => res.json())\n      .then(setData)\n      .catch(setError)\n      .finally(() => setLoading(false));\n  }, [url]);\n\n  return { data, loading, error };\n}",
        },
        {
            "instruction": "Create a React modal component with open and close props.",
            "input": "",
            "output": "export function Modal({ isOpen, onClose, children }) {\n  if (!isOpen) return null;\n  return (\n    <div className='modal-backdrop' onClick={onClose}>\n      <div className='modal' onClick={(e) => e.stopPropagation()}>\n        <button onClick={onClose}>Close</button>\n        {children}\n      </div>\n    </div>\n  );\n}",
        },
        {
            "instruction": "Create a React navbar component with navigation links.",
            "input": "",
            "output": "export function Navbar() {\n  return (\n    <nav>\n      <a href='/'>Home</a>\n      <a href='/products'>Products</a>\n      <a href='/login'>Login</a>\n    </nav>\n  );\n}",
        },
        {
            "instruction": "Create a React protected route wrapper component.",
            "input": "Redirect to /login if user is not authenticated.",
            "output": "import { Navigate } from 'react-router-dom';\n\nexport function ProtectedRoute({ isAuthenticated, children }) {\n  if (!isAuthenticated) {\n    return <Navigate to='/login' replace />;\n  }\n  return children;\n}",
        },
        {
            "instruction": "Create a React form with basic client-side validation.",
            "input": "Email is required.",
            "output": "import { useState } from 'react';\n\nexport function ContactForm() {\n  const [email, setEmail] = useState('');\n  const [error, setError] = useState('');\n\n  function handleSubmit(event) {\n    event.preventDefault();\n    if (!email) {\n      setError('Email is required');\n      return;\n    }\n    setError('');\n  }\n\n  return (\n    <form onSubmit={handleSubmit}>\n      <input value={email} onChange={(e) => setEmail(e.target.value)} />\n      {error && <p>{error}</p>}\n      <button type='submit'>Send</button>\n    </form>\n  );\n}",
        },
        {
            "instruction": "Create a React loading spinner component.",
            "input": "",
            "output": "export function Spinner() {\n  return <div className='spinner' aria-label='Loading' />;\n}",
        },
        {
            "instruction": "Create a React theme context provider.",
            "input": "",
            "output": "import { createContext, useContext, useState } from 'react';\n\nconst ThemeContext = createContext();\n\nexport function ThemeProvider({ children }) {\n  const [theme, setTheme] = useState('light');\n  return (\n    <ThemeContext.Provider value={{ theme, setTheme }}>\n      {children}\n    </ThemeContext.Provider>\n  );\n}\n\nexport function useTheme() {\n  return useContext(ThemeContext);\n}",
        },
        {
            "instruction": "Create a React debounced search input component.",
            "input": "",
            "output": "import { useEffect, useState } from 'react';\n\nexport function SearchInput({ onSearch }) {\n  const [value, setValue] = useState('');\n\n  useEffect(() => {\n    const timer = setTimeout(() => onSearch(value), 300);\n    return () => clearTimeout(timer);\n  }, [value, onSearch]);\n\n  return <input value={value} onChange={(e) => setValue(e.target.value)} placeholder='Search...' />;\n}",
        },
        {
            "instruction": "Create a React pagination component.",
            "input": "",
            "output": "export function Pagination({ page, totalPages, onPageChange }) {\n  return (\n    <div>\n      <button disabled={page <= 1} onClick={() => onPageChange(page - 1)}>Previous</button>\n      <span>{page} / {totalPages}</span>\n      <button disabled={page >= totalPages} onClick={() => onPageChange(page + 1)}>Next</button>\n    </div>\n  );\n}",
        },
        {
            "instruction": "Create a React todo list component.",
            "input": "",
            "output": "import { useState } from 'react';\n\nexport function TodoList() {\n  const [todos, setTodos] = useState([]);\n  const [text, setText] = useState('');\n\n  function addTodo() {\n    setTodos([...todos, { id: Date.now(), text, done: false }]);\n    setText('');\n  }\n\n  return (\n    <div>\n      <input value={text} onChange={(e) => setText(e.target.value)} />\n      <button onClick={addTodo}>Add</button>\n      <ul>{todos.map((todo) => <li key={todo.id}>{todo.text}</li>)}</ul>\n    </div>\n  );\n}",
        },
        {
            "instruction": "Create a React dropdown select component.",
            "input": "",
            "output": "export function Dropdown({ value, options, onChange }) {\n  return (\n    <select value={value} onChange={(e) => onChange(e.target.value)}>\n      {options.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}\n    </select>\n  );\n}",
        },
        {
            "instruction": "Create a React table component for user data.",
            "input": "Columns: id, name, email.",
            "output": "export function UserTable({ users }) {\n  return (\n    <table>\n      <thead>\n        <tr><th>ID</th><th>Name</th><th>Email</th></tr>\n      </thead>\n      <tbody>\n        {users.map((user) => (\n          <tr key={user.id}>\n            <td>{user.id}</td>\n            <td>{user.name}</td>\n            <td>{user.email}</td>\n          </tr>\n        ))}\n      </tbody>\n    </table>\n  );\n}",
        },
        {
            "instruction": "Create a React API service module for products.",
            "input": "",
            "output": "export async function getProducts() {\n  const response = await fetch('/api/products');\n  if (!response.ok) throw new Error('Failed to fetch products');\n  return response.json();\n}\n\nexport async function createProduct(data) {\n  const response = await fetch('/api/products', {\n    method: 'POST',\n    headers: { 'Content-Type': 'application/json' },\n    body: JSON.stringify(data),\n  });\n  if (!response.ok) throw new Error('Failed to create product');\n  return response.json();\n}",
        },
        {
            "instruction": "Create a React component that shows an error message when API fetch fails.",
            "input": "",
            "output": "import { useEffect, useState } from 'react';\n\nexport function ProductList() {\n  const [products, setProducts] = useState([]);\n  const [error, setError] = useState('');\n\n  useEffect(() => {\n    fetch('/api/products')\n      .then((res) => {\n        if (!res.ok) throw new Error('Request failed');\n        return res.json();\n      })\n      .then(setProducts)\n      .catch((err) => setError(err.message));\n  }, []);\n\n  if (error) return <p>Error: {error}</p>;\n  return <ul>{products.map((p) => <li key={p.id}>{p.name}</li>)}</ul>;\n}",
        },
        {
            "instruction": "Create a React file upload input component.",
            "input": "",
            "output": "export function FileUpload({ onFileSelect }) {\n  return (\n    <input\n      type='file'\n      onChange={(event) => onFileSelect(event.target.files?.[0] || null)}\n    />\n  );\n}",
        },
    ],
    "nextjs.jsonl": [
        {
            "instruction": "Create a Next.js App Router home page component.",
            "input": "",
            "output": "export default function HomePage() {\n  return (\n    <main>\n      <h1>Welcome to My App</h1>\n      <p>Built with Next.js App Router.</p>\n    </main>\n  );\n}",
        },
        {
            "instruction": "Create a Next.js App Router API route that returns JSON status.",
            "input": "Path: app/api/health/route.ts",
            "output": "import { NextResponse } from 'next/server';\n\nexport async function GET() {\n  return NextResponse.json({ status: 'ok' });\n}",
        },
        {
            "instruction": "Create a Next.js API route for creating products.",
            "input": "Accept JSON body with name and price.",
            "output": "import { NextResponse } from 'next/server';\n\nexport async function POST(request) {\n  const body = await request.json();\n  if (!body.name || body.price == null) {\n    return NextResponse.json({ message: 'Invalid payload' }, { status: 400 });\n  }\n  return NextResponse.json({ id: 1, ...body }, { status: 201 });\n}",
        },
        {
            "instruction": "Create a Next.js server component that fetches posts.",
            "input": "",
            "output": "export default async function PostsPage() {\n  const response = await fetch('https://jsonplaceholder.typicode.com/posts', { cache: 'no-store' });\n  const posts = await response.json();\n\n  return (\n    <ul>\n      {posts.slice(0, 5).map((post) => <li key={post.id}>{post.title}</li>)}\n    </ul>\n  );\n}",
        },
        {
            "instruction": "Create a Next.js client component login form.",
            "input": "",
            "output": "'use client';\n\nimport { useState } from 'react';\n\nexport default function LoginForm() {\n  const [email, setEmail] = useState('');\n  const [password, setPassword] = useState('');\n\n  return (\n    <form>\n      <input value={email} onChange={(e) => setEmail(e.target.value)} placeholder='Email' />\n      <input type='password' value={password} onChange={(e) => setPassword(e.target.value)} placeholder='Password' />\n      <button type='submit'>Login</button>\n    </form>\n  );\n}",
        },
        {
            "instruction": "Create a Next.js dynamic route page for product details.",
            "input": "Route: app/products/[id]/page.tsx",
            "output": "export default function ProductPage({ params }) {\n  return (\n    <main>\n      <h1>Product {params.id}</h1>\n    </main>\n  );\n}",
        },
        {
            "instruction": "Create a Next.js root layout with shared navigation.",
            "input": "",
            "output": "export default function RootLayout({ children }) {\n  return (\n    <html lang='en'>\n      <body>\n        <nav>\n          <a href='/'>Home</a>\n          <a href='/products'>Products</a>\n        </nav>\n        {children}\n      </body>\n    </html>\n  );\n}",
        },
        {
            "instruction": "Add metadata to a Next.js page.",
            "input": "",
            "output": "export const metadata = {\n  title: 'Products',\n  description: 'Browse all products',\n};\n\nexport default function ProductsPage() {\n  return <h1>Products</h1>;\n}",
        },
        {
            "instruction": "Create a Next.js middleware that protects /dashboard routes.",
            "input": "",
            "output": "import { NextResponse } from 'next/server';\n\nexport function middleware(request) {\n  const token = request.cookies.get('token');\n  if (!token && request.nextUrl.pathname.startsWith('/dashboard')) {\n    return NextResponse.redirect(new URL('/login', request.url));\n  }\n  return NextResponse.next();\n}",
        },
        {
            "instruction": "Create a Next.js server action for submitting a contact form.",
            "input": "",
            "output": "'use server';\n\nexport async function submitContact(formData) {\n  const email = formData.get('email');\n  const message = formData.get('message');\n  if (!email || !message) {\n    throw new Error('Email and message are required');\n  }\n  return { success: true };\n}",
        },
        {
            "instruction": "Create a Next.js loading UI component.",
            "input": "File: app/products/loading.tsx",
            "output": "export default function Loading() {\n  return <p>Loading products...</p>;\n}",
        },
        {
            "instruction": "Create a Next.js error UI component.",
            "input": "File: app/products/error.tsx",
            "output": "'use client';\n\nexport default function Error({ error, reset }) {\n  return (\n    <div>\n      <p>Something went wrong: {error.message}</p>\n      <button onClick={() => reset()}>Try again</button>\n    </div>\n  );\n}",
        },
        {
            "instruction": "Create a Next.js API route that reads environment variables safely.",
            "input": "",
            "output": "import { NextResponse } from 'next/server';\n\nexport async function GET() {\n  return NextResponse.json({ appEnv: process.env.APP_ENV || 'development' });\n}",
        },
        {
            "instruction": "Create a Next.js fetch call with revalidation.",
            "input": "Revalidate every 60 seconds.",
            "output": "export default async function Page() {\n  const res = await fetch('https://example.com/api/data', { next: { revalidate: 60 } });\n  const data = await res.json();\n  return <pre>{JSON.stringify(data, null, 2)}</pre>;\n}",
        },
        {
            "instruction": "Create a Next.js redirect from /old-products to /products.",
            "input": "",
            "output": "import { redirect } from 'next/navigation';\n\nexport default function OldProductsPage() {\n  redirect('/products');\n}",
        },
        {
            "instruction": "Create a Next.js notFound page behavior for missing product.",
            "input": "",
            "output": "import { notFound } from 'next/navigation';\n\nexport default function ProductPage({ product }) {\n  if (!product) {\n    notFound();\n  }\n  return <h1>{product.name}</h1>;\n}",
        },
        {
            "instruction": "Create a Next.js page that uses next/link for navigation.",
            "input": "",
            "output": "import Link from 'next/link';\n\nexport default function HomePage() {\n  return (\n    <main>\n      <Link href='/products'>Go to products</Link>\n    </main>\n  );\n}",
        },
        {
            "instruction": "Create a protected Next.js API route that checks Authorization header.",
            "input": "",
            "output": "import { NextResponse } from 'next/server';\n\nexport async function GET(request) {\n  const auth = request.headers.get('authorization');\n  if (auth !== 'Bearer dev-token') {\n    return NextResponse.json({ message: 'Unauthorized' }, { status: 401 });\n  }\n  return NextResponse.json({ secret: 'protected-data' });\n}",
        },
        {
            "instruction": "Create a Next.js route handler for updating a resource by id.",
            "input": "",
            "output": "import { NextResponse } from 'next/server';\n\nexport async function PUT(request, { params }) {\n  const body = await request.json();\n  return NextResponse.json({ id: params.id, ...body });\n}",
        },
        {
            "instruction": "Create a Next.js page that renders product cards from fetched data.",
            "input": "",
            "output": "export default async function ProductsPage() {\n  const products = [{ id: 1, name: 'Laptop', price: 999 }];\n  return (\n    <div>\n      {products.map((product) => (\n        <div key={product.id}>\n          <h2>{product.name}</h2>\n          <p>${product.price}</p>\n        </div>\n      ))}\n    </div>\n  );\n}",
        },
    ],
    "debugging.jsonl": [
        {
            "instruction": "Fix this Python bug and explain the issue.",
            "input": "def get_average(numbers):\n    total = 0\n    for i in range(len(numbers) + 1):\n        total += numbers[i]\n    return total / len(numbers)",
            "output": "Issue: range(len(numbers) + 1) causes an off-by-one error and accesses an invalid index, raising IndexError.\n\nFixed code:\n\ndef get_average(numbers):\n    if not numbers:\n        return 0\n    return sum(numbers) / len(numbers)",
        },
        {
            "instruction": "Fix this Python bug and explain the issue.",
            "input": "items = []\nprint(items[0])",
            "output": "Issue: The list is empty, so accessing index 0 raises IndexError.\n\nFixed code:\n\nitems = []\nif items:\n    print(items[0])\nelse:\n    print('No items available')",
        },
        {
            "instruction": "Fix this Python bug and explain the issue.",
            "input": "user = {'name': 'Alice'}\nprint(user['email'])",
            "output": "Issue: The dictionary does not contain the key 'email', so KeyError is raised.\n\nFixed code:\n\nuser = {'name': 'Alice'}\nprint(user.get('email', 'No email found'))",
        },
        {
            "instruction": "Fix this Python bug and explain the issue.",
            "input": "def greet(name):\n    print(name.upper())\n\ngreet(None)",
            "output": "Issue: name is None, so calling upper() raises AttributeError.\n\nFixed code:\n\ndef greet(name):\n    if name:\n        print(name.upper())\n    else:\n        print('Hello, guest')",
        },
        {
            "instruction": "Fix this JavaScript bug and explain the issue.",
            "input": "async function loadUsers() {\n  const response = fetch('/api/users');\n  return response.json();\n}",
            "output": "Issue: fetch returns a Promise, but await is missing before response.json() can be used correctly.\n\nFixed code:\n\nasync function loadUsers() {\n  const response = await fetch('/api/users');\n  return response.json();\n}",
        },
        {
            "instruction": "Fix this Python bug and explain the issue.",
            "input": "def add_item(item, items=[]):\n    items.append(item)\n    return items",
            "output": "Issue: Using a mutable default argument shares the same list across function calls.\n\nFixed code:\n\ndef add_item(item, items=None):\n    if items is None:\n        items = []\n    items.append(item)\n    return items",
        },
        {
            "instruction": "Fix this JavaScript bug and explain the issue.",
            "input": "for (var i = 0; i < 3; i++) {\n  setTimeout(() => console.log(i), 100);\n}",
            "output": "Issue: var is function-scoped, so all timeouts log the final value of i.\n\nFixed code:\n\nfor (let i = 0; i < 3; i++) {\n  setTimeout(() => console.log(i), 100);\n}",
        },
        {
            "instruction": "Fix this Python bug and explain the issue.",
            "input": "def divide(a, b):\n    return a / b\n\nprint(divide(10, 0))",
            "output": "Issue: Division by zero raises ZeroDivisionError.\n\nFixed code:\n\ndef divide(a, b):\n    if b == 0:\n        raise ValueError('Cannot divide by zero')\n    return a / b",
        },
        {
            "instruction": "Fix this JavaScript bug and explain the issue.",
            "input": "const data = JSON.parse('{name: \"Alice\"}');",
            "output": "Issue: The string is not valid JSON because keys must be quoted.\n\nFixed code:\n\nconst data = JSON.parse('{\"name\": \"Alice\"}');",
        },
        {
            "instruction": "Fix this Python bug and explain the issue.",
            "input": "numbers = [1, 2, 3]\nprint(numbers[3])",
            "output": "Issue: The valid indexes are 0 to 2, so numbers[3] raises IndexError.\n\nFixed code:\n\nnumbers = [1, 2, 3]\nprint(numbers[-1])",
        },
        {
            "instruction": "Fix this JavaScript bug and explain the issue.",
            "input": "const user = null;\nconsole.log(user.name);",
            "output": "Issue: user is null, so accessing user.name throws TypeError.\n\nFixed code:\n\nconst user = null;\nconsole.log(user?.name ?? 'Unknown user');",
        },
        {
            "instruction": "Fix this Python bug and explain the issue.",
            "input": "def factorial(n):\n    return n * factorial(n - 1)",
            "output": "Issue: The function has no base case, causing infinite recursion.\n\nFixed code:\n\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
        },
        {
            "instruction": "Fix this React bug and explain the issue.",
            "input": "useEffect(() => {\n  fetchData();\n}, []);",
            "output": "Issue: If fetchData is recreated on every render and used inside useEffect, stale closures or missing dependencies can cause bugs.\n\nFixed code:\n\nuseEffect(() => {\n  fetchData();\n}, [fetchData]);",
        },
        {
            "instruction": "Fix this Python bug and explain the issue.",
            "input": "result = ''\nfor char in ['a', 'b', 'c']:\n    result = result + char + ', '\nprint(result)",
            "output": "Issue: This works but adds an extra comma and space at the end.\n\nFixed code:\n\nresult = ', '.join(['a', 'b', 'c'])\nprint(result)",
        },
        {
            "instruction": "Fix this JavaScript bug and explain the issue.",
            "input": "if (value = 10) {\n  console.log('Value is ten');\n}",
            "output": "Issue: = is assignment, not comparison, so the condition is always truthy.\n\nFixed code:\n\nif (value === 10) {\n  console.log('Value is ten');\n}",
        },
        {
            "instruction": "Fix this Python bug and explain the issue.",
            "input": "def is_even(n):\n    if n % 2 = 0:\n        return True\n    return False",
            "output": "Issue: = is assignment; comparison requires ==.\n\nFixed code:\n\ndef is_even(n):\n    return n % 2 == 0",
        },
        {
            "instruction": "Fix this Node.js bug and explain the issue.",
            "input": "app.get('/users', async (req, res) => {\n  const users = db.getUsers();\n  res.json(users);\n});",
            "output": "Issue: db.getUsers() is async but not awaited, so the response may receive a Promise instead of data.\n\nFixed code:\n\napp.get('/users', async (req, res) => {\n  const users = await db.getUsers();\n  res.json(users);\n});",
        },
        {
            "instruction": "Fix this Python bug and explain the issue.",
            "input": "def find_user(users, user_id):\n    for user in users:\n        if user['id'] == user_id:\n            print(user['name'])",
            "output": "Issue: The function prints the name but does not return anything, which makes reuse harder.\n\nFixed code:\n\ndef find_user(users, user_id):\n    for user in users:\n        if user['id'] == user_id:\n            return user\n    return None",
        },
        {
            "instruction": "Fix this JavaScript bug and explain the issue.",
            "input": "const total = [1, 2, 3].reduce((sum, n) => sum + n);",
            "output": "Issue: reduce requires an initial value or the first element handling can be confusing for empty arrays.\n\nFixed code:\n\nconst total = [1, 2, 3].reduce((sum, n) => sum + n, 0);",
        },
        {
            "instruction": "Fix this FastAPI bug and explain the issue.",
            "input": "@app.get('/users/{id}')\ndef get_user(id: int):\n    return users[id]",
            "output": "Issue: users[id] assumes list index equals user id, which is usually incorrect and can raise IndexError.\n\nFixed code:\n\n@app.get('/users/{id}')\ndef get_user(id: int):\n    for user in users:\n        if user['id'] == id:\n            return user\n    raise HTTPException(status_code=404, detail='User not found')",
        },
    ],
}


def write_batches() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for filename, records in BATCHES.items():
        path = RAW_DIR / filename
        with path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Wrote {len(records)} records to {path}")


if __name__ == "__main__":
    write_batches()
