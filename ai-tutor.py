import sys
import os  
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

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

class Evaluation(BaseModel):
    is_correct: bool = Field(description="True if the student's answer is correct.")
    reason: str = Field(description="Why the answer passed or failed the check.")


class TutorSystem:
    def __init__(self, api_key):
        self.hint_counter = 0
        self.waiting_for_answer = False
        self.question = ""
        self.chat_history = []

        try:
            self.model = ChatGroq(
                model="openai/gpt-oss-120b", 
                temperature=0.3,
                api_key=api_key,
                max_tokens=2000,
                streaming=True
            )

        except Exception as e:
            print(f"Failed to initialize Groq client: {e}")
            sys.exit(1)

        self.tools = [math, science, history, english]
        self.model_with_tools = self.model.bind_tools(self.tools)
        self.question_verification = self.model.with_structured_output(StudentQuestionEvaluation)
        self.evaluator_model = self.model.with_structured_output(Evaluation)
    
    def chat(self, user_input: str):
        if not self.waiting_for_answer:
            response = self.generate(user_input=user_input)
        else:
            response = self.evaluate_student(student_answer=user_input)
        
        self.update_history(user_input, response)
        return response
        
    def update_history(self, user_message, ai_message):
        self.chat_history.append(HumanMessage(content=user_message))
        self.chat_history.append(AIMessage(content=ai_message))

    def generate(self, user_input: str):
        check_prompt = f"Analyze this input: '{user_input}'. Is this a legitimate question about Math, Science, History, or English Grammar?"
        
        try:
            question_verify = self.question_verification.invoke(check_prompt)
            if not question_verify.is_correct:
                return "I am a specialized tutor for Math, Science, History, and English Grammar only. I cannot answer questions or topics outside my expertise. What would you like to learn within my 4 subjects?"
        except Exception:
            is_valid = False

        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are the best Master Tutor Agent. SYSTEM PROMPT: SOCRATIC INSTRUCTOR.
            You are STRICTLY RESTRICTED to exactly four subjects:
            1. Mathematics
            2. Science
            3. History
            4. English Grammar
            
            If a student asks about anything outside of this four subjects especially coding related question in any language or
            anything then refuse the request and tell the student that I have only knowledge of 4 subjects only maths science history and english grammer.
             
            Follow these Behavioral Guidelines:
            1. Identify the student's primary subject area.
            2. Retrieve verified information through tools.
            3. Provide perfect and focused explanations.
            4. Answer only maths, science, history and english grammer questions. If student asks about anything else then tell the student your knowledge is in these 4 subjects only please ask questions from these subjects.
            5. Maintain a supportive, professional, and encouraging teaching style. 
            6. At last generate one question to solve the concept with the proper example without any hint as a question: question.   
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        self.agent = create_tool_calling_agent(llm=self.model, tools=self.tools, prompt=prompt)

        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=False)

        response = self.agent_executor.invoke({
            "input": user_input,
            "chat_history": self.chat_history
        })
        
        self.question = response["output"]
        self.waiting_for_answer = "Question:" or "question:" in self.question
        return response["output"]
    
    def evaluate_student(self, student_answer):
        eval_prompt = f"Question: {self.question}\nAnswer: {student_answer} Is this correct?"
        
        try:
            evaluation = self.evaluator_model.invoke(eval_prompt)
        except Exception:
            evaluation = Evaluation(is_correct=False, reason="Error")

        if evaluation.is_correct:
            self.waiting_for_answer = False
            self.hint_counter = 0
            return f"Correct Answer!\n{evaluation.reason} \n\n What do you want to learn new?"
        
        self.hint_counter += 1
        
        if self.hint_counter == 1:
            hint_instruction = "Give a HARD hint. Just a tiny clue. Do not say the answer."
        elif self.hint_counter == 2:
            hint_instruction = "Give a MEDIUM hint. Give the user the right logic. Do not say the answer."
        elif self.hint_counter == 3:
            hint_instruction = "Give an EASY hint. Give almost all the steps and nearby answer. Do not say the answer."
        else:
            self.waiting_for_answer = False
            self.hint_counter = 0

            solution_prompt = f"Question: {self.question} Give the full correct answer with proper explanation and Do NOT ask any new questions."
            solution = self.model.invoke(solution_prompt).content

            return f"Solution: {solution} \n\n What do you want to learn new?"

        hint_prompt = f"Question: {self.question}\nStudent said: {student_answer} (Wrong).\nInstruction: {hint_instruction}"
        hint = self.model.invoke(hint_prompt).content

        return f"Not the right answer. {hint}"

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
            user_input = input("[Student]: ")
            if user_input == "0":
                print("\n[System]: Thank you for using our AI Tutor System")
                print("\n=== Exiting Tutor System. ===")
                break
            answer = tutor_system.chat(user_input=user_input)
            print(f"\n[Tutor]: {answer}\n")
        except KeyboardInterrupt:
            print("\n=== Exiting Tutor System. ===")
            break
