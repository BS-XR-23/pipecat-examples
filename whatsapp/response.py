from random import choice



GREETING_RESPONSES = [
    "Hello! I am Jarvis. How can I help you today?",
    "Hi there. My name is Jarvis. What can I do for you?",
    "Welcome to ABB Bank support. I am Jarvis. How may I assist you?",
]


HUMAN_HANDOFF = [
    "Sure — I’ll connect you with a human representative.\n\n📞 ABB Bank Support Desk: +880-800-ABB-HELP",
    "No problem. A human agent can assist you further.\n\n📞 You can reach us at +880-800-ABB-HELP",
    "Alright, I’ll refer you to a support specialist.\n\n📞 Please call: +880-800-ABB-HELP",
    "I understand. Let me guide you to a human representative.\n\n📞 ABB Support: +880-800-ABB-HELP",
    "You can speak directly with our support team.\n\n📞 +880-800-ABB-HELP"
]


THANK_YOU_RESPONSES = [
    "You're welcome 😊 Let me know if you need anything else.",
    "Happy to help! If you need anything else, I'm here.",
    "No problem at all 👍 Feel free to ask if anything comes up.",
]

FALLBACK_RESPONSES = [
    "I’m not fully sure I understood that. Could you rephrase it?",
    "Sorry, I didn’t quite get that. Can you clarify?",
    "I want to make sure I help correctly — could you explain a bit more?",
]

SMALL_TALK_RESPONSES = [
    "I'm here to help you with your banking needs 😊",
    "How can I assist you with your account today?",
    "Let me know what you need help with 👍",
]


BALANCE_MISSING_PHONE = [
    "Please provide your registered phone number so I can check your balance.",
    "I’ll need your registered phone number to continue.",
    "Could you share your account phone number?",
]

BALANCE_SUCCESS = [
    "Your account balance is {balance} {currency}.",
    "You currently have {balance} {currency} in your account.",
]


USER_INFO_MISSING_PHONE = [
    "Please provide your registered phone number.",
    "I need your phone number to fetch your profile.",
]

USER_INFO_SUCCESS = [
    "Name: {name}, Email: {email}",
    "Here are your details → Name: {name}, Email: {email}",
]


RAG_FALLBACK = [
    "Here’s what I found based on your query:",
    "I looked into it — here’s the relevant information:",
    "Based on available information:",
]


TICKET_CONFIRM_INTENT = [
    "I understand you're facing an issue. Do you want to create a support ticket or connect you to a human agent?",
    "Would you like me to raise a ticket or connect you with a human agent?",
    "Should I proceed with creating a support ticket or connect you with a human agent?"
]


TICKET_ASK_ISSUE = [
    "Please briefly describe your issue.",
    "Tell me what problem you're facing.",
    "Kindly explain your issue so I can assist you."
]


TICKET_ASK_PHONE = [
    "Please provide your registered phone number for verification.",
    "I need your account phone number to proceed.",
    "Kindly share your registered phone number."
]


TICKET_ACK_RECEIVED = [
    "Got it. We have received your issue.",
    "Thank you. Your request has been recorded.",
    "Noted. I’m processing your issue now..."
]


TICKET_PROCESSING = [
    "Your ticket is being created...",
    "We are processing your request...",
    "Please wait while we generate your support ticket..."
]


TICKET_FINAL = [
    "Your ticket has been created successfully.",
    "Support ticket generated. Our team will contact you soon.",
    "Done. Your issue has been logged with support.",
]


GREETING_RESPONSES_HOTEL = [
    "Welcome! I'm your hotel booking assistant. How can I help you today?",
    "Hello! I'm here to help you find the perfect room. What do you need?",
    "Hi there! Ready to assist with your stay. What can I do for you?",
]

SMALL_TALK_RESPONSES_HOTEL = [
    "You're welcome! Let me know if there's anything else I can help with.",
    "Happy to help! Is there anything else you need?",
    "Of course! Feel free to ask anytime.",
]

SEARCH_ASK_CHECKIN = [
    "Sure! What's your check-in date? (e.g. 2025-06-01)",
    "Let's find you a room. When would you like to check in?",
    "What date are you planning to check in?",
]

SEARCH_ASK_CHECKOUT = [
    "Great! And what's your check-out date?",
    "Got it. When will you be checking out?",
    "And when do you plan to leave?",
]

SEARCH_ASK_GUESTS = [
    "How many guests will be staying?",
    "How many people will be checking in?",
    "And how many guests are there?",
]

BOOKING_ASK_ROOM = [
    "Which room would you like to book? (Standard, Deluxe, or Suite)",
    "Which room would you prefer?",
    "Please let me know which room you'd like to reserve.",
]

BOOKING_ASK_NAME = [
    "Could I have your full name for the booking?",
    "What name should I put on the reservation?",
    "Please provide your full name.",
]

BOOKING_ASK_PHONE = [
    "Could I have your phone number to complete the booking?",
    "What's your contact number?",
    "Please provide a phone number for this reservation.",
]

BOOKING_SUCCESS = [
    "Your booking is confirmed!",
    "All set! Your reservation is confirmed.",
    "Booking confirmed successfully!",
]

CANCEL_ASK_BOOKING_ID = [
    "Please provide your booking ID to proceed.",
    "What's the booking ID you'd like to cancel?",
    "Could you share your booking ID?",
]

CANCEL_ASK_PHONE = [
    "Please provide the phone number associated with this booking for verification.",
    "Could I have the phone number used for this booking?",
]

CANCEL_CONFIRM = [
    "Are you sure you want to cancel booking {booking_id}? Reply yes to confirm.",
    "Just to confirm — cancel booking {booking_id}? Reply yes to proceed.",
]

CANCEL_SUCCESS = [
    "Done! Booking {booking_id} has been cancelled successfully.",
    "Your booking {booking_id} has been cancelled.",
]

CHECK_ASK_BOOKING_ID = [
    "Please provide your booking ID and I'll look it up.",
    "What's your booking ID?",
    "Could you share your booking reference ID?",
]

HUMAN_HANDOFF_HOTEL = [
    "I'll connect you with our front desk team right away.",
    "Sure, I'm transferring you to a human representative now.",
]


def pick(options, **kwargs):
    """
    Random response picker with optional formatting.
    Prevents crashes if keys are missing.
    """
    try:
        return choice(options).format(**kwargs)
    except Exception:
        return choice(options)