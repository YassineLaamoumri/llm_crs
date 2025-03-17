You are provided with a raw transcript from an audio file of an e-commerce service call. The conversation is in Arabic and includes various dialects. Your task is to process this input and produce a final output that not only cleans and structures the conversation but also extracts actionable insights to help maximize customer satisfaction and confirmation rates. Follow the steps below precisely:

1. **Standardize and Clean the Conversation:**
   - Convert the entire conversation into clear, standard Modern Standard Arabic.
   - Remove dialect-specific slang, informal expressions, or unclear phrases.
   - Preserve the context, meaning, and any nuances that could affect interpretation.

2. **Identify and Label Speakers:**
   - Identify and label speakers using the labels "مندوب خدمة العملاء" (Customer Service Representative) and "عميل" (Customer).
   - Retain any specific names or roles mentioned; if none are provided, use the default labels.
   - Do not include any timestamps.

3. **Structure the Conversation for a Knowledge Base:**
   - Extract and summarize key information from the call. Your structured summary must include the following variables:
     - **Call ID:** A unique identifier for the call.
     - **Issue Summary:** A concise overview of the customer's inquiry or problem.
     - **Customer's Details and Request:** Specific details of the customer's questions, requirements, or complaints.
     - **Agent’s Response and Action Items:** A summary of the solutions, guidance, or actions provided by the agent.
     - **Outcome:** The final result of the conversation, indicated as either "Confirmation" (e.g., purchase or agreement) or "Rejection".
     - **Outcome Reason:** A brief explanation of why the outcome was as it was.
     - **Upsell Information:** Details of any upsell attempts, including the nature of the upsell, the pitch details, and its result.
     - **Additional Metadata:** Critical insights including:
         - **Call Duration:** (if available)
         - **Overall Sentiment:** (e.g., positive, negative, neutral)
         - **Keywords:** Extract significant keywords from the conversation.
         - **Contextual Factors:** Any external factors or conversation patterns that may have influenced the outcome.
         - **Correlation Patterns:** Any observed patterns that correlate with higher confirmation rates.
     - **Agent Guidance:** Actionable recommendations based on the call analysis:
         - **Actions to Do:** Specific actions the agent should take to satisfy the customer.
         - **Actions to Avoid:** Actions that should be avoided to prevent customer dissatisfaction.

4. **Translate the Content to English:**
   - Provide a complete English translation of both:
     - The cleaned, speaker-labeled Arabic conversation.
     - The structured knowledge base entry.
   - Ensure that the translation is accurate and retains the original context and detailed insights.

5. **Final Output Format:**
   - Your final answer must be a JSON object with the following keys:
     - `"cleaned_arabic_conversation"`: Contains the cleaned and speaker-labeled conversation in Arabic.
     - `"knowledge_base_entry"`: Contains the structured knowledge base summary in Arabic, including all specified variables.
     - `"english_translation"`: Contains the English translation of both the conversation and the knowledge base entry, organized as follows:
         - `"Cleaned Conversation"`: The English translation of the cleaned conversation.
         - `"Knowledge Base Entry"`: The English translation of the structured summary, with these fields:
             - **Call ID**
             - **Issue Summary**
             - **Customer's Request**
             - **Agent's Response**
             - **Outcome** (only "Confirmation" or "Rejection")
             - **Outcome Reason**
             - **Actions to Do**
             - **Actions to Avoid**
             - **Upsell Information**
             - **Additional Metadata**

Ensure that your output is well-organized, follows a clear, step-by-step internal processing approach, and includes only the final JSON output without any extraneous commentary.

Example JSON Structure:
{
  "cleaned_arabic_conversation": "<Your cleaned and labeled Arabic conversation here>",
  "knowledge_base_entry": {
      "Call ID": "<Unique identifier>",
      "Issue Summary": "<Concise summary of the customer's problem>",
      "Customer's Request": "<Detailed customer inquiry>",
      "Agent's Response": "<Summary of agent's actions>",
      "Outcome": "<Confirmation or Rejection>",
      "Outcome Reason": "<Brief explanation of the outcome>",
      "Upsell Information": "<Details on upsell attempt and result>",
      "Additional Metadata": {
         "Call Duration": "<Call duration if available>",
         "Overall Sentiment": "<Sentiment analysis>",
         "Keywords": "<List of significant keywords>",
         "Contextual Factors": "<Any external influencing factors>",
         "Correlation Patterns": "<Patterns observed with confirmation rates>"
      },
      "Agent Guidance": {
         "Actions to Do": "<Recommended actions for the agent>",
         "Actions to Avoid": "<Actions that should be avoided>"
      }
  },
  "english_translation": {
    "Cleaned Conversation": "<English translation of the conversation>",
    "Knowledge Base Entry": {
       "Call ID": "<Unique identifier>",
       "Issue Summary": "<English summary>",
       "Customer's Request": "<English customer details>",
       "Agent's Response": "<English summary of agent's actions>",
       "Outcome": "<English outcome>",
       "Outcome Reason": "<English explanation>",
       "Actions to Do": "<English recommended actions>",
       "Actions to Avoid": "<English actions to avoid>",
       "Upsell Information": "<English upsell details>",
       "Additional Metadata": "<English additional insights>"
    }
  }
}
