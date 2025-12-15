#!/usr/bin/env python3
"""
ACE Browser Agent (WITH ACE)

Browser automation agent with learning capabilities using ACE framework.
Compare this with baseline_browser_use.py to see ACE's value.
"""

import asyncio
from typing import List, Dict
from dotenv import load_dotenv

from browser_use import Agent, Browser, ChatOpenAI

from ace import (
    LiteLLMClient,
    Generator,
    Reflector,
    Curator,
    OnlineAdapter,
    Sample,
    TaskEnvironment,
    EnvironmentResult,
    Playbook,
)
load_dotenv()

from utils import print_history_details



class BrowserUseEnvironment(TaskEnvironment):
    """Environment that evaluates browser automation performance."""

    def __init__(self, headless: bool = True, model: str = "gpt-4o-mini"):
        self.headless = headless
        self.model = model

    def evaluate(self, sample: Sample, generator_output):
        """Run browser automation and evaluate the result."""

        task = sample.context
        action_plan = generator_output.final_answer

        assert isinstance(action_plan, list), "Action plan must be a list"

        num_steps = len(action_plan)
        action_plan = "\n".join(action_plan)

        browser_use_prompt = f"""
        {task}

        Follow these steps:
        {action_plan}
        """

        # Run browser automation
        history = asyncio.run(self._run_browser_task(browser_use_prompt))

        #Get a list of strings, each string is a step taken by the browser use agent
        steps_taken = len(history.extracted_content()) if hasattr(history, "extracted_content") else 0

        #Output of the browser use agent
        final_result = history.final_result() if hasattr(history, "final_result") else ""

        #Things that can be used to evaluate the performance of the browser use agent
        is_done = history.is_done() if hasattr(history, "is_done") else False
        is_successful = history.is_successful() if hasattr(history, "is_successful") else False
        has_errors = history.has_errors() if hasattr(history, "has_errors") else False
        number_of_steps = history.number_of_steps() if hasattr(history, "number_of_steps") else 0

        # Print all history information in a nice format just to see what information is available
        # print_history_details(history)
      
        feedback = f"""
        The task was {"" if is_done else "not "} finished.
        The task was {"" if is_successful else "not "}successful.
        The browser use agent had {"" if has_errors else "no "}errors.
        Browser use agent took {number_of_steps} steps.

        The steps taken were: 
        {"\n".join(steps_taken)}

        The final result was: {final_result}
        """

        status = "ERROR"
        if is_successful:
            status = "SUCCESS"
        else:
            status = "ERROR"


        return EnvironmentResult(
            feedback=feedback,
            ground_truth=None,  # No ground truth available for form filling
            metrics={
                "success": success,
                "efficient": efficient,
                "steps": number_of_steps,
                "status": status,
            }
        )


    async def run_browser_task(browser_use_prompt: str, model: str = "gpt-4o-mini", headless: bool = True):
        """Run browser task without any learning."""
        
        
        browser = None
        try:
            # Start browser
            browser = Browser(headless=headless)
            await browser.start()

            # Create agent with basic task (no learning, no strategy optimization)
            llm = ChatOpenAI(model=model, temperature=0.0)

            agent = Agent(
                task=task,
                llm=llm,
                browser=browser,
                max_actions_per_step=5,
                max_steps=15
            )

            # Run with timeout
            history = await asyncio.wait_for(agent.run(), timeout=240.0)


            return history
        except asyncio.TimeoutError:
            try: 
                number_of_steps = history.number_of_steps() if hasattr(history, "number_of_steps") else 0
            except:
                number_of_steps = 25  # max_steps if we can't determine
            return {"is_done": False, "is_successful": False, "has_errors": True, "number_of_steps": number_of_steps , "final_result": "Timeout"}
        except Exception as e:
            try:
                number_of_steps = history.number_of_steps() if hasattr(history, "number_of_steps") else 0
            except:
                number_of_steps = 0
            return {"is_done": False, "is_successful": False, "has_errors": True, "number_of_steps": 0, "final_result": str(e)}

        finally:
            if browser:
                try:
                    await browser.stop()
                except:
                    pass


def main(task_file: str = "task1_flight_search.txt"):
    """Main function - browser automation with ACE learning.
    
    Args:
        task_file: Path to the task file containing the browser task description.
                  Defaults to "task1_flight_search.txt".
    """

    print("\n🤖 ACE Browser Agent (WITH ACE)")
    print("✨ Learning enabled - improves after each task")
    print("=" * 40)
    

    print("\n🔄 Starting browser task with learning...\n")

    # Read task from file
    with open(task_file, "r") as f:
        task_content = f.read()
    task_content = "Task:\n\n" + task_content

    results = []

    llm = LiteLLMClient(model="gpt-4o-mini", temperature=0.7)

    adapter = OnlineAdapter(
        playbook=Playbook(),
        generator=Generator(llm),
        reflector=Reflector(llm),
        curator=Curator(llm),
        max_refinement_rounds=2,
    )

    environment = BrowserUseEnvironment(
        headless=False,
        model="gpt-4o-mini",
    )

    
    question = f"""
    If you were a browser use agent, what would you do to fullfil the following task?

    How would your step by step action plan look like for this browser use task?
    Single steps should be atomic and self-contained.
    The action plan should be a list of steps.
    Single steps could be things like:
    - Click on a button
    - Fill out a form
    - Navigate to a website
    - Read a value from the screen
    - etc.

    Provide the plan as an overall answer in the final_answer field as a list of steps.
    Example: "final_answer": ["Step 1: Click on the \"next\" button", "Step 2: Fill out the Email field", "Step 3: Fill out the Password field", "Step 4: Click on the login button"]
    """


    samples = []
    for i in range(5):
        samples.append(Sample(
            question=question,
            ground_truth="SUCCESS",
            context=task_content
        ))


    results = adapter.run(samples, environment)

    result = asyncio.run(run_browser_task(task=task_content, headless=False))
    results.append(result)

    # Show final results
    print("=" * 40)
    print("📊 Results:")


    # Summary
    successful = sum(1 for r in results if r['success'])
    total_steps = sum(r['steps'] for r in results)
    avg_steps = total_steps / len(results) if results else 0

    print(f"\n✅ Success rate: {successful}/{len(results)} ({100*successful/len(results):.1f}%)")
    print(f"⚡ Average steps: {avg_steps:.1f}")
    print(f"✨ Learning enabled - improves after each task")

    print(f"\n💡 Compare with: python examples/browser-use/baseline_browser_use.py")
    print(f"   Baseline has no learning - same performance every time")


if __name__ == "__main__":
    main()