using Microsoft.ML;
using Microsoft.ML.Data;

public class HousingPricePrediction
{
    [ColumnName("Score")]
    public float Price { get; set; }
}