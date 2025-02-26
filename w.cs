public class Solution
{
    public IList<IList<int>> ThreeSum(int[] nums)
    {
        IList<IList<int>> answer = new IList<IList<int>();
        int length = nums.getLength();
        for (i = 0; i < length; i++)
        {
            for (j = 0; j < length; j++)
            {
                for (k = 0; k < length; k++)
                {
                    int[] out = [nums[i], nums[j], nums[k]];
                    out.sort();
                    if (i != j and j != k and k != i and nums[i] +nums[j] + nums[k] == 0 and out not in answer){ 
                        answer.append(out)
                    }

                }

            }
                
                        
        }
            
        return answer
    }
}