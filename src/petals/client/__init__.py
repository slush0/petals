from petals.client.inference_session import InferenceSession
from petals.client.remote_model import (
    DistributedBloomConfig,
    DistributedBloomForCausalLM,
    DistributedBloomForSequenceClassification,
    DistributedBloomModel,
)
# if os.getenv("PETALS_LLAMA"):
from petals.client.remote_model_llama import (
    DistributedLLaMAConfig,
    DistributedLLaMAForCausalLM,
    DistributedLLaMAModel,
)
from petals.client.remote_sequential import RemoteSequential, RemoteTransformerBlock
from petals.client.routing.sequence_manager import RemoteSequenceManager
from petals.client.routing.spending_policy import NoSpendingPolicy, SpendingPolicyBase
