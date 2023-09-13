✨States & Environment & Action
=============================

States
------
**Definition:**

As previously mentioned, we utilize SOP to operate the autonomous agent. Our SOP reasoning graph is composed of various States. These states play distinct roles, contributing to the entire system. We've developed a straightforward state, the SIMPLIST state, primarily based on LLM.

**Attributes & Examples:**

- Basic codes for State class are as follows:

  .. code:: python

     class State:
         """
         Sub-scenes of role activities, responsible for storing the tasks that each role needs to do
         """
         def __init__(self, **kwargs):
             self.next_states = {}
             self.name = kwargs["name"]
     
             self.environment_prompt = (
                 kwargs["environment_prompt"] if "environment_prompt" in kwargs else ""
             )
     
             self.roles = kwargs["roles"] if "roles" in kwargs else [0]
             self.begin_role = (
                 kwargs["begin_role"] if "begin_role" in kwargs else self.roles[0]
             )
             self.begin_query = kwargs["begin_query"] if "begin_query" in kwargs else None
     
             self.is_begin = True
     
             self.summary_prompt = (
                 kwargs["summary_prompt"] if "summary_prompt" in kwargs else None
             )
             self.current_role = self.begin_role
             self.components = (
                 self.init_components(kwargs["agent_states"])
                 if "agent_states" in kwargs
                 else {}
             )
             self.index = (
                 self.roles.index(self.begin_role) if self.begin_role in self.roles else 0
             )
             self.chat_nums = 0

  Part of the attributes are shown below:

  - name: The only tag of the state.
  - components: Contains components with different roles in this state. When different agents perform tasks in this state, they will obtain components from them according to their own roles.
  - environment_prompt: For the description of the current state, it will be added in front of the system prompt of each agent in the state.
  - summary_prompt: Prompt initiated from the system. For detailed information, please turn to Component page.
  - begin_role & begin_query: Role and query of the agents which is set at the beginning of the conversation.
  - next_states: Relations between states. Extraordinarily useful when the state graph is relatively sophisticated.

**Methods:**

Our states provide one single method, namely init_components, which is shown as follows:

**init_components**

The init_components method receives various types of components, and then classify and place them. Basic codes are omitted.

Environment
-----------
**Definition:**

Apparently, every autonomous agent should adjust to different circumstances, thus changing their chatting style and information immediately. To help manage their self-evolution, we established the memory mode to guide its behaviors. As its name shows, the memory module stores the whole chatting history of the particular agent. To edit and compile its contents and update the memory in time, we use the Environment module to guide its behavior.

**Attributes & Examples:**

Basic codes of an Environment module are as follows:

.. code:: python

   class Environment:
       def __init__(self, config) -> None:
           self.shared_memory = {"long_term_memory": [], "short_term_memory": None}
           self.agents = None

           self.summary_system_prompt = {}
           self.summary_last_prompt = {}
           self.environment_prompt = {}
           self.environment_type = config["environment_type"] if "environment_type" in config else "cooperate"
           self.current_chat_history_idx = 0
           self.LLMs = {}

           # Initialize the summary method for each state
           for state_name, state_dict in config["states"].items():
               if state_name != "end_state":
                   self.summary_system_prompt[state_name] = (
                       state_dict["summary_system_prompt"]
                       if "summary_system_prompt" in state_dict
                       else eval(Default_environment_summary_system_prompt)
                   )

                   self.summary_last_prompt[state_name] = (
                       state_dict["summary_last_prompt"]
                       if "summary_last_prompt" in state_dict
                       else eval(Default_environment_summary_last_prompt)
                   )

                   self.environment_prompt[state_name] = (
                       state_dict["environment_prompt"]
                       if "environment_prompt" in state_dict
                       else " "
                   )
                   LLM_type = (
                       state_dict["LLM_type"] if "LLM_type" in state_dict else "OpenAI"
                   )
                   if LLM_type == "OpenAI":
                       if "LLM" in state_dict:
                           self.LLMs[state_name] = OpenAILLM(**state_dict["LLM"])
                       else:
                           self.LLMs[state_name] = OpenAILLM(model="gpt-3.5-turbo-16k-0613", temperature=0.3,
                                                              log_path=f"logs/{state_name}")
           self.roles_to_names = None
           self.names_to_roles = None

  Part of the attributes are shown below:

  - LLM: As is aforementioned, our autonomous agents are based on LLM. This attribute receives the tag of a certain type of LLM and invokes it.

**Methods:**

**summary:**

The summary method receives the current chatting history, and then summarizes the situation in the current environment every once in a while.

.. code:: python

   def summary(self, current_state):
       """
       Summarize the situation in the current environment every once in a while
       """
       MAX_CHAT_HISTORY = eval(os.environ["MAX_CHAT_HISTORY"])
       current_state_name = current_state.name

       query = self.shared_memory["long_term_memory"][-1].content
       relevant_history = get_relevant_history(
           query,
           self.shared_memory["long_term_memory"][:-1],
           self.shared_memory["chat_embeddings"][:-1],
       )

       relevant_history = Memory.get_chat_history(relevant_history)
       chat_history = Memory.get_chat_history(
           self.shared_memory["long_term_memory"][-MAX_CHAT_HISTORY + 1 :]
       )
       summary = self.shared_memory["short_term_memory"]

       # system prompt = environment prompt + current memory + system prompt
       # current_memory = summary + chat history + relevant history
       current_memory = eval(Environment_summary_memory)
       environment_prompt = self.environment_prompt[current_state_name]
       summary_system_prompt = self.summary_system_prompt[current_state_name]

       environment_summary_system_prompt = eval(Environment_summary_system_prompt)
       response = self.LLMs[current_state_name].get_response(None, environment_summary_system_prompt, stream=False)
       return response

**update_memory:**

The update_memory method updates memory immediately, enabling the agent to adjust to current circumstance.

.. code:: python

   def update_memory(self, memory, current_state):
       """
       update chat embbedings and long term memory,short term memory,agents long term memory
       """
       MAX_CHAT_HISTORY = eval(os.environ["MAX_CHAT_HISTORY"])
       self.shared_memory["long_term_memory"].append(memory)
       current_embedding = get_embedding(memory.content)
       if "chat_embeddings" not in self.shared_memory:
           self.shared_memory["chat_embeddings"] = current_embedding
       else:
           self.shared_memory["chat_embeddings"] = torch.cat(
               [self.shared_memory["chat_embeddings"], current_embedding], dim=0
           )
       if len(self.shared_memory["long_term_memory"]) % MAX_CHAT_HISTORY == 0:
           summary = self.summary(current_state)
           self.shared_memory["short_term_memory"] = summary

       self.agents[memory.send_name].update_memory(memory)

**_observe:**

The _observe method helps the agent obtain the memories it needs to reply from the environment, including related memories and new memories.

.. code:: python

   def _observe(self, agent):
       MAX_CHAT_HISTORY = eval(os.environ["MAX_CHAT_HISTORY"])
       current_state = agent.current_state
       current_role = agent.state_roles[current_state.name]
       current_component_dict = current_state.components[current_role]

       # cooperative: Sharing information between different states ; competitive: No information is shared between different states
       current_chat_history_idx = self.current_chat_history_idx if self.environment_type == "competitive" else 0
       current_long_term_memory = self.shared_memory["long_term_memory"][current_chat_history_idx:]
       current_chat_embeddings = self.shared_memory["chat_embeddings"][current_chat_history_idx:]

       # relevant_memory
       query = current_long_term_memory[-1].content

       relevant_memory = get_relevant_history(
           query,
           current_long_term_memory[:-1],
           current_chat_embeddings[:-1],
       )
       relevant_memory = Memory.get_chat_history(relevant_memory, agent.name)

       relevant_memory = eval(Agent_observe_relevant_memory)
       agent.relevant_memory = relevant_memory

       # get chat history from new conversation
       conversations = self._get_agent_new_memory(agent, current_long_term_memory)

       # memory = relevant_memory + summary + history + query
       query = current_long_term_memory[-1]
       current_memory = eval(Agent_observe_memory)

       return {"role": "user", "content": current_memory}

Action
------
**Definition:**

The basic unit for each Agent to interact

**Attributes & Examples:**

Basic codes of an Action module are as follows:

.. code:: python

   class Action:
       """
       The basic action unit of agent
       """
       def __init__(self, **kwargs):
           self.response = None
           self.is_user = False
           self.res_dict = {}
           self.name = ""
           self.role = ""
           for key, value in kwargs.items():
               setattr(self, key, value)

**Methods:**

**process:**

The current action will be processed, and the response required by the user will be obtained.

.. code:: python

   def process(self):
       """
       processing action
       Return: memory(Memory)
       """
       response = self.response
       send_name = self.name
       send_role = self.role
       all = ""
       for res in response:
           all += res
       parse = f"{send_name}:"

       # The third person in the dialogue was deleted.
       while parse in all:
           index = all.index(parse) + len(parse)
           all = all[index:]
       if not self.is_user:
           print(f"{send_name}({send_role}):{all}")
       memory = Memory(send_role, send_name, all)
       return memory