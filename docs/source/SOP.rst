📋Standard Operation Procedure (SOP) System
===========================================

Definition
-----------
A **Standard Operating Procedure (SOP)** is a reasoning graph that consists of a set of step-by-step instructions outlining how to execute a specific task or process. Overall, the SOP System enables users to communicate with different agents simultaneously or create virtual cases, allowing agents to interact with each other.

Demonstrations & Remarks
------------------------

Our SOP System provides several distinct functions, including **init_series**, **transit**, **route**, **load_date**, and **send_memory**, which are shown as follows:

Init_series
~~~~~~~~~~~

**SOP_init:**

.. code:: python

    class SOP:
        """
        Responsible for managing the operational processes of all agents
        """
        
        # SOP should have args: "states", "relations", "root"
        
        def __init__(self, **kwargs):
            self.controller_dict = {}
            LLM_type = kwargs["LLM_type"] if "LLM_type" in kwargs else "OpenAI"
            if LLM_type == "OpenAI":
                self.LLM = (
                    OpenAILLM(**kwargs["LLM"])
                    if "LLM" in kwargs
                    else OpenAILLM(model="gpt-3.5-turbo-16k-0613", temperature=0.3, log_path="logs/god")
                )

            self.states = {}
            self.init_states(kwargs["states"])
            self.init_relation(kwargs["relations"])
            for state_name, states_dict in kwargs["states"].items():
                if state_name != "end_state" and "controller" in states_dict:
                    self.controller_dict[state_name] = states_dict["controller"]

            self.user_names = kwargs["user_names"] if "user_names" in kwargs else []
            self.root = self.states[kwargs["root"]]
            self.current_state = self.root
            self.finish_state_name = (
                kwargs["finish_state_name"]
                if "finish_state_name" in kwargs
                else "end_state"
            )
            self.roles_to_names = None
            self.names_to_roles = None
            self.finished = False

    # Remark:
    # Part of the attributes of the whole SOP System is shown as follows:
    # - LLM: As is aforementioned, our autonomous agents are based on LLM. This attribute receives the tag of a certain type of LLM and invokes it.
    # - states: The fundamental attribute of an SOP. It stores the information of various agents, including their data and background, which helps run the whole reasoning graph.
    # - name & role & relation: Basic attributes between various types of agents. Act as tags of the agents.
    # - controller: To manage our states' activation order, we introduce the controller module. By sending instructions and orders, our controller allocates tasks for each Node and comes up with a proper system order.

States_init
~~~~~~~~~~~

.. code:: python

    def init_states(self, states_dict):
        for state_name, state_dict in states_dict.items():
            self.states[state_name] = State(**state_dict)

    # Remark:
    # - states_dict: Components of various types of agents. Whenever the node is activated by users, it will at first select the proper Agent to start its Component.
    # - Please refer to the Environment part for detailed definitions and explanations on other aforementioned attributes.

Relation_init
~~~~~~~~~~~~

.. code:: python

    def init_relation(self, relations):
        for state_name, state_relation in relations.items():
            for idx, next_state_name in state_relation.items():
                self.states[state_name].next_states[idx] = self.states[next_state_name]

    # Remark:
    # - Please refer to the States part for detailed definitions and explanations.

Transit
~~~~~~

.. code:: python

    def transit(self, chat_history, **kwargs):
        """
        Determine the next state based on the current situation
        Return: 
        next_state (State): the next state
        """
        # 如果是单一循环节点，则一直循环即可
        # If it is a single loop node, just keep looping
        if len(self.current_state.next_states) == 1:
            next_state = "0"
            
        # 否则则需要 controller 去判断进入哪一节点
        # Otherwise, the controller needs to determine which node to enter.
        else:
            current_state = self.current_state
            controller_dict = self.controller_dict[current_state.name]
            relevant_history = kwargs["relevant_history"]
            
            max_chat_nums = controller_dict["max_chat_nums"] if "max_chat_nums" in controller_dict else 1000
            if current_state.chat_nums >= max_chat_nums:
                return self.current_state.next_states["1"]
            
            # 否则则让 controller 判断是否结束
            # Otherwise, let the controller judge whether to end
            judge_system_prompt = controller_dict["judge_system_prompt"]
            environment_prompt = eval(Get_environment_prompt) if current_state.environment_prompt else ""
            transit_system_prompt = eval(Transit_system_prompt)
            
            judge_last_prompt = controller_dict["judge_last_prompt"]
            transit_last_prompt = eval(Transit_last_prompt)
            
            environment = kwargs["environment"]
            environment_summary = environment.shared_memory["short_term_memory"]
            chat_history_message = Memory.get_chat_history(chat_history)
            query = chat_history[-1].get_query()
            
            chat_messages = [
                {
                    "role": "user",
                    "content": eval(Transit_message)
                }
            ]
            
            extract_words = controller_dict["judge_extract_words"] if "judge_extract_words" in controller_dict else "end"

            response = self.LLM.get_response(
                chat_messages, transit_system_prompt, transit_last_prompt, stream=False, **kwargs
            )
            next_state = (
                response if response.isdigit() else extract(response, extract_words)
            )
            
            # 如果没有 parse 出来则继续循环
            # If no parse comes out, continue looping
            if not next_state.isdigit():
                next_state = "0"
            
        next_state = self.current_state.next_states[next_state]
        return next_state

    # Remark:
    # The Transit method judges which state the SOP graph should run based on the current situation. It can also invoke the controller module to automatically determine which state should be called for.

Route
~~~~~

.. code:: python

    def route(self, chat_history, **kwargs):
        """
        Determine the role that needs action based on the current situation
        Return: 
        current_agent (Agent): the next act agent
        """
        
        agents = kwargs["agents"]
        
        # 知道进入哪一状态后开始分配角色，如果该状态下只有一个角色则直接分配给他
        # Start assigning roles after knowing which state you have entered. If there is only one role in that state, assign it directly to him.
        if len(self.current_state.roles) == 1:
            next_role = self.current_state.roles[0]
        
        # 否则 controller 进行分配
        # Otherwise the controller determines
        else:
            relevant_history = kwargs["relevant_history"]
            controller_type = (
                self.controller_dict[self.current_state.name]["controller_type"]
                if "controller_type" in self.controller_dict[self.current_state.name]
                else "rule"
            )

            # 如果是 rule 控制器，则交由 LLM 进行分配角色
            # If controller type is rule, it is left to LLM to assign roles.
            if controller_type == "rule":
                controller_dict = self.controller_dict[self.current_state.name]
                
                call_last_prompt = controller_dict["call_last_prompt"] if "call_last_prompt" in controller_dict else ""
                
                allocate_prompt = ""
                roles = list(set(self.current_state.roles))
                for role in roles:
                    allocate_prompt += eval(Allocate_component)
                    
                call_system_prompt = controller_dict["call_system_prompt"] if "call_system_prompt" in controller_dict else ""
                environment_prompt = eval(Get_environment_prompt) if self.current_state.environment_prompt else ""    
                # call_system_prompt + environment + allocate_prompt 
                call_system_prompt = eval(Call_system_prompt)
                
                query = chat_history[-1].get_query()
                last_name = chat_history[-1].send_name
                # last_prompt: note + last_prompt + query
                call_last_prompt = eval(Call_last_prompt)
                
                chat_history_message = Memory.get_chat_history(chat_history)
                # Intermediate historical conversation records
                chat_messages = [
                    {
                        "role": "user",
                        "content": eval(Call_message),
                    }
                ]

                extract_words = controller_dict["call_extract_words"] if "call_extract_words" in controller_dict else "end"

                response = self.LLM.get_response(
                    chat_messages, call_system_prompt, call_last_prompt, stream=False, **kwargs
                )

                # get next role
                next_role = extract(response, extract_words)

            # Speak in order
            elif controller_type == "order":
                # If there is no begin role, it will be given directly to the first person.
                if not self.current_state.current_role:
                    next_role = self.current_state.roles[0]
                # otherwise first
                else:
                    self.current_state.index += 1
                    self.current_state.index =  (self.current_state.index) % len(self.current_state.roles)
                    next_role = self.current_state.roles[self.current_state.index]
            # random speak
            elif controller_type == "random":
                next_role = random.choice(self.current_state.roles)
            
        # 如果下一角色不在，则随机挑选一个
        # If the next character is not available, pick one at random    
        if next_role not in self.current_state.roles:
            next_role = random.choice(self.current_state.roles)
            
        self.current_state.current_role = next_role 
        
        next_agent = agents[self.roles_to_names[self.current_state.name][next_role]]
        
        return next_agent

    # Remark:
    # The Route method judges which role of agent should be invoked based on the current situation. The Route method first gets the state of the controller, then makes actions based on the type of the controller.

Next
~~~~

.. code:: python

    def next(self, environment, agents):
        """
        Determine the next state and the agent that needs action based on the current situation
        """
        
        # 如果是第一次进入该状态
        # If it is the first time to enter this state
        
        if self.current_state.is_begin:
            agent_name = self.roles_to_names[self.current_state.name][self.current_state.begin_role]
            agent = agents[agent_name]
            return self.current_state, agent
    
    
        # get relevant history
        query = environment.shared_memory["long_term_memory"][-1].content
        relevant_history = get_relevant_history(
            query,
            environment.shared_memory["long_term_memory"][:-1],
            environment.shared_memory["chat_embeddings"][:-1],
        )
        relevant_history = Memory.get_chat_history(relevant_history)
        
        
        
        next_state = self.transit(
            chat_history=environment.shared_memory["long_term_memory"][
                environment.current_chat_history_idx :
            ],
            relevant_history=relevant_history,
            environment=environment,
        )
        # 如果进入终止节点，则直接终止
        # If you enter the termination node, terminate directly
        if next_state.name == self.finish_state_name:
            self.finished = True
            return None, None

        self.current_state = next_state
        
        # 如果是首次进入该节点且有开场白，则直接分配给开场角色
        # If it is the first time to enter the state and there is a begin query, it will be directly assigned to the begin role.
        if self.current_state.is_begin and self.current_state.begin_role:
            agent_name = self.roles_to_names[self.current_state.name][self.current_state.begin_role]
            agent = agents[agent_name]
            return self.current_state, agent
           

        next_agent = self.route(
            chat_history=environment.shared_memory["long_term_memory"][
                environment.current_chat_history_idx :
            ],
            agents=agents,
            relevant_history=relevant_history,
        )

        return self.current_state, next_agent

    # Remark:
    # The Next method determines the next state and the role that needs action based on the current situation. Detailed remarks are added to the codes.

Examples
--------

We provide diverse SOP Systems of various types of Agents. Get to know in our QuickStart part! 🌟