import sys
import os 
import uuid 
import logging
import psycopg2
from dotenv import load_dotenv
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename="logs/report.log",
                    format='%(asctime)s %(levelname)s: %(message)s',
                    filemode='a')

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

class DatabaseManager:
    def __init__(self):
        self.host = os.environ.get("DB_HOST")
        self.database = os.environ.get("DB_NAME")
        self.user = os.environ.get("DB_USER")
        self.password = os.environ.get("DB_PASS")

        try:
            self.connection = psycopg2.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password
            )
            self.create_table()
            logger.info("Successfully connected to PostgreSQL database.")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            print(f"\n[Warning]: Could not connect to PostgreSQL.\n[Error]: {e}\n")
            self.connection = None
        
    def create_table(self):
        query = """
        CREATE TABLE IF NOT EXISTS student_report (
            id SERIAL PRIMARY KEY,
            session_id VARCHAR(50) NOT NULL,
            subject VARCHAR(50) NOT NULL,
            question TEXT NOT NULL,
            result VARCHAR(50) NOT NULL,
            hints INTEGER NOT NULL,
            timestamp TIMESTAMP NOT NULL DEFAULT NOW()
        );
        """
        with self.connection.cursor() as cursor:
            cursor.execute(query)
            self.connection.commit()
    
    def record_chat(self, session_id, subject, question, result, hints_used):
        query = """
        INSERT INTO student_report (session_id, subject, question, result, hints)
        VALUES (%s, %s, %s, %s, %s);
        """
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, (session_id, subject, question, result, hints_used))
                self.connection.commit()
        except Exception as e:
            logger.error(f"Failed to insert record: {e}")
    
    def get_session_details(self, session_id):
        query = "SELECT subject, question, result, hints FROM student_report WHERE session_id = %s;"

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query,(session_id,))
                user_data = cursor.fetchall()
                return user_data
        except Exception as e:
            logger.error(f"Failed to fetch: {e}")
            return []


@tool
def math(query: str):
    """Invokes [math] to resolve mathematical calculations and rules."""
    return "Math context: Data retrieved from mathematical ruleset."

@tool
def science(query: str):
    """Invokes [science] to retrieve from science database."""
    return "Science context: Data retrieved from science database."

@tool
def history(query: str):
    """Invokes [history] to retrieve from historical timeline."""
    return "History context: Data retrieved from historical timeline."

@tool
def english(query: str):
    """Invokes [english] to retrieve from english grammar ruleset."""
    return "English context: Data retrieved from english grammar ruleset."


class StudentQuestionEvaluation(BaseModel):
    is_correct: bool = Field(
        description="True ONLY if the question is strictly about Math, Science, History, or English. False if it asks for programming, coding or unrelated topics."
    )
    subject: Literal["Math", "Science", "History", "English", "Other"] = Field(
        description="The subject of the student's question."
    )
    resume_subject: bool = Field(
        description="True if the user is simply asking to return to a previous subject(e.g.'history', 'math', 'science'). False if they are asking a specific new question."
    )


class Evaluation(BaseModel):
    is_correct: bool = Field(description="True if the student's answer is correct.")
    reason: str = Field(description="Why the answer passed or failed the check.")


class UserReply(BaseModel):
    is_new_question: bool = Field(
        description="True if the user is asking a new question, seeking clarification, or changing the topic. False if the user is attempting to answer the tutor's current question."
    )


class TutorSystem:
    def __init__(self, api_key):
        self.session_id = str(uuid.uuid4())
        self.db = DatabaseManager()

        self.hint_counter = {
            "Math": 0,
            "Science": 0,
            "History": 0,
            "English": 0,
            "Other": 0
        }

        self.current_subject = "Other"
        
        self.active_questions = {
            "Math": "", 
            "Science": "", 
            "History": "", 
            "English": "", 
            "Other": ""
        }
        
        self.waiting_for_answer = {
            "Math": False, 
            "Science": False, 
            "History": False, 
            "English": False, 
            "Other": False
        }

        self.chat_history = []

        try:
            self.model = ChatGroq(
                model="openai/gpt-oss-120b", 
                temperature=0.3,
                api_key=api_key,
                max_tokens=4000,
            )

        except Exception as e:
            print(f"Failed to initialize Groq client: {e}")
            sys.exit(1)

        self.tools = [math, science, history, english]
        self.question_verification = self.model.with_structured_output(StudentQuestionEvaluation)
        self.evaluator_model = self.model.with_structured_output(Evaluation)
        self.user_reply = self.model.with_structured_output(UserReply)
    
    def chat(self, user_input: str):
        logger.info(f"message received from user: {user_input}")

        switched_subj = self.handle_subject_switch(user_input)

        if switched_subj:
            self.current_subject = switched_subj
            if self.waiting_for_answer[switched_subj]:
                response = f"[System: Resuming {switched_subj} - Question: {self.active_questions[switched_subj]}"
            else:
                response = f"[System: Switched to {switched_subj}] What would you like to ask?"
            
            self.update_history(user_input, response)
            return response
 
        is_new = False
        if self.waiting_for_answer[self.current_subject]:
            reply_prompt = (
                f"Tutor asked: '{self.active_questions[self.current_subject]}'\n"
                f"Student replied: '{user_input}'\n"
                "Analyze the student's reply. Is the student asking a new question or changing the topic?"
                "Your ONLY job is to output the requested structured data. "
                "DO NOT explain your reasoning. DO NOT generate any conversational text."
            )
            
            try:
                check_is_question = self.user_reply.invoke(reply_prompt)
                if check_is_question.is_new_question:
                    logger.info(f"new question arrived from user: {user_input}")
                    is_new = True
            except Exception as e:
                print(f"[System Error]: {e}")

        if not self.waiting_for_answer[self.current_subject] or is_new:
            response = self.generate(user_input=user_input)
        else:
            response = self.evaluate_student(student_answer=user_input)
        
        self.update_history(user_input, response)

        return response
    
    def handle_subject_switch(self, user_input):
        input = user_input.strip().lower()

        subjects = {
            "math": "Math",
            "maths": "Math",
            "science": "Science",
            "history": "History",
            "english": "English"
        }

        if input in subjects:
            return subjects[input]
        
        return None

    def update_history(self, user_message, ai_message):
        self.chat_history.append(HumanMessage(content=user_message))
        self.chat_history.append(AIMessage(content=ai_message))

    def generate(self, user_input: str):
        logger.info(f"question switched or created new")
        check_prompt = f"Analyze this input: '{user_input}'. Is this a legitimate question about Math, Science, History, or English Grammar? Also identify the exact subject and if the user is just asking to resume."
        
        try:
            question_verify = self.question_verification.invoke(check_prompt)
            if not question_verify.is_correct and not question_verify.resume_subject:
                return "I am a specialized tutor for Math, Science, History, and English Grammar only. I cannot answer questions or topics outside my expertise. What would you like to learn within my 4 subjects?"

            new_subject = question_verify.subject
            logger.info(f"New Subject: {new_subject}")
            if self.waiting_for_answer[new_subject]:
                logger.info(f"waiting for answer")
                if question_verify.resume_subject:
                    logger.info(f"resume new subject")
                    self.current_subject = new_subject
                    return f"Resuming the {new_subject}\nQuestion: {self.active_questions[new_subject]}"
                else:
                    self.hint_counter[new_subject] = 0
                    self.waiting_for_answer[new_subject] = False

            self.current_subject = question_verify.subject
            logger.info(f"new subject {self.current_subject}")
            self.hint_counter[self.current_subject] = 0
        
        except Exception as e:
            print(f"[System Error]: {e}")

        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are the best Master Tutor Agent. SYSTEM PROMPT: SOCRATIC INSTRUCTOR.
            You are STRICTLY RESTRICTED to exactly four subjects:
            1. Mathematics
            2. Science
            3. History
            4. English Grammar
            
            If a student asks about anything outside of this four subjects especially coding related question in any language or anything then 
            refuse the request and tell the student that I have only knowledge of 4 subjects only maths science history and english grammar.
            
            Follow these Behavioral Guidelines:
            1. Identify the student's primary subject area.
            2. Retrieve verified information through tools.
            3. Provide perfect and focused explanations.
            4. Answer only maths, science, history and english grammar questions. If student asks about anything else then tell the student your knowledge is in these 4 subjects only please ask questions from these subjects.
            5. Maintain a supportive, professional, and encouraging teaching style. 
            6. If the student has answered the previous question and if it is wrong then just give the hint not the right answer. 
            7. At last generate one question to solve the concept with the proper example without any hint as a question: question.  
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        self.agent = create_tool_calling_agent(llm=self.model, tools=self.tools, prompt=prompt)

        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=False)

        try:
            response = self.agent_executor.invoke({
                "input": user_input,
                "chat_history"  : self.chat_history
            })
            output = response["output"]
            self.active_questions[self.current_subject] = output

            self.waiting_for_answer[self.current_subject] = "question" in output.lower()
            return response["output"]
        
        except Exception as e:
            print(f"[System Error]: {e}")
            return "Please resend the message after wait time"
    
    def evaluate_student(self, student_answer):
        active_que = self.active_questions[self.current_subject]
        eval_prompt = f"Question: {active_que}\nAnswer: {student_answer} Is this correct?"
        
        try:
            evaluation = self.evaluator_model.invoke(eval_prompt)
        except Exception:
            evaluation = Evaluation(is_correct=False, reason="Error")

        if evaluation.is_correct:
            self.db.record_chat(
                session_id=self.session_id, 
                subject=self.current_subject, 
                question=active_que, 
                result="Correct Answer", 
                hints_used=self.hint_counter[self.current_subject]
            )

            self.waiting_for_answer[self.current_subject] = False
            self.hint_counter[self.current_subject] = 0
            self.active_questions[self.current_subject] = ""

            return f"Correct Answer!\n{evaluation.reason} \n\nWhat do you want to learn new?"
        
        self.hint_counter[self.current_subject] += 1
        
        if self.hint_counter[self.current_subject] == 1:
            hint_instruction = "Give a HARD hint. Just a tiny clue. Do not say the answer."
            self.db.record_chat(
                session_id=self.session_id, 
                subject=self.current_subject, 
                question=active_que, 
                result="Wrong Answer", 
                hints_used=self.hint_counter[self.current_subject]
            )

        elif self.hint_counter[self.current_subject] == 2:
            hint_instruction = "Give a MEDIUM hint. Give the user the right logic. Do not say the answer."
            self.db.record_chat(
                session_id=self.session_id, 
                subject=self.current_subject, 
                question=active_que, 
                result="Wrong Answer", 
                hints_used=self.hint_counter[self.current_subject]
            )

        elif self.hint_counter[self.current_subject] == 3:
            hint_instruction = "Give an EASY hint. Give almost all the steps and nearby answer. Do not say the answer."
            self.db.record_chat(
                session_id=self.session_id, 
                subject=self.current_subject, 
                question=active_que, 
                result="Wrong Answer", 
                hints_used=self.hint_counter[self.current_subject]
            )

        else:
            self.db.record_chat(
                session_id=self.session_id, 
                subject=self.current_subject, 
                question=active_que, 
                result="Wrong Answer", 
                hints_used=self.hint_counter[self.current_subject]
            )

            self.waiting_for_answer[self.current_subject] = False
            self.hint_counter[self.current_subject] = 0
            self.active_questions[self.current_subject] = ""

            solution_prompt = f"Question: {active_que} Give the full correct answer with proper explanation and Do NOT ask any new questions."
            
            try:
                solution = self.model.invoke(solution_prompt).content
            except Exception as e:
                print(f"[System Error]: {e}")
            self.active_questions[self.current_subject] = ""
            return f"Solution: {solution} \n\n What do you want to learn new?"

        hint_prompt = f"Question: {self.active_questions[self.current_subject]}\nStudent said: {student_answer} (Wrong).\nInstruction: {hint_instruction}"
        
        try:
            hint = self.model.invoke(hint_prompt).content
        except Exception as e:
                print(f"[System Error]: {e}")

        return f"{hint}"
    
    def generate_report(self): 
        data = self.db.get_session_details(session_id=self.session_id)
        if not data:
            return "No questions found!"
        
        report_prompt = f"""
        You are an expert Educational Evaluator. Based on the following questions and answers , generate a performance report for the student.

        Student Metrics:
        {data}
        Requirements:
        1. Summarize their performance across the subjects they attempted.
        2. Provide feedback on their reliance on hints.
        3. If there are multiple questions and the question is repeated then consider ther question as 1 question and count as its attempts.
        4. Calculate a final numerical score out of 100 based on their 'Result' if Result is Correct Answer then the user has given the answer 
        correct if Wrong answer then user has given wrong answer if less hints used then that is good and more hints used then that is not good. 
        5. If same question has multiple hints that means user is attempting to answer but giving the wrong answers.
        6. Make the report professional, constructive, and visually clear
        7. Do NOT ask any follow-up questions or any other questions.
        """
        
        try:
            report = self.model.invoke(report_prompt).content
            return report
        except Exception as e:
            print(f"[System Error]: {e}")
            return f"Failed to generate report: {e}"
    

if __name__ == "__main__":
    api_key = os.environ.get("GROQ_API_KEY")
    
    if not api_key:
        print("No API Key Found. Exiting...")
        sys.exit(1)
    
    tutor_system = TutorSystem(api_key=api_key)
    print("=== AI Tutor System ===\n")
    while True:
        try:
            print("[System]: For exiting the conversation press 0 and enter.\n")
            print("[System]: For generating the report write 'report' and enter.\n")
            print(f"Subject: {tutor_system.current_subject}")
            print(f"Hint: {tutor_system.hint_counter[tutor_system.current_subject]}")
            print(f"Waiting for answer: {tutor_system.waiting_for_answer}")

            user_input = input("[Student]: ")
            if user_input == "0":
                print("\n[System]: Thank you for using our AI Tutor System")
                print("\n=== Exiting Tutor System. ===")
                break
            elif user_input == "report":
                print("\n[System]: Generating Student Report Using our AI Tutor System")
                report = tutor_system.generate_report()
                print(report)
                continue

            answer = tutor_system.chat(user_input=user_input)
            print(f"\n[Tutor]: {answer}\n")
        except KeyboardInterrupt:
            print("\n=== Exiting Tutor System. ===")
            break
