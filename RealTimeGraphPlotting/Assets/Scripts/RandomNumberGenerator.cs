using System.Collections;
using System.Collections.Generic;
using UnityEngine;



public class RandomNumberGenerator : MonoBehaviour
{
    // Start is called before the first frame update

    void Start()
    {
    
    }

    // Update is called once per frame
    void Update()
    {
      
      
      float num = Random.Range(0,1000);
      //float num2 = Random.Range(-num, num);
      
      Graph_Creator.instance.UpdateGraph(num);
      
      
      
    }
}
