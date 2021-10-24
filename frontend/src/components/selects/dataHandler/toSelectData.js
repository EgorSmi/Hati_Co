export function toSelectData(arr_objects, field1, field2){
    /*  
    Обработка массива обьекта данных в вид для select
    На вход подаем:
    arr_objects - массив наших обьектов на обработку
    field1 - название поля которое из наших обьектов будет == value 
    field2 - название поля которое из наших обьектов будет == label

    arr_objects = [
            {
                "id": 1,
                "name": "Mazda",
                "image": "/img1"
            },
        ]   
    field1 = id
    field2 = name

    function return  [
        {
            "value": 1,
            "label": "Mazda"
        },
    ]

    Если поля field1 || field2 нет в arr_objects functuion return ( [null] )
    
    */
    let dataEditToSelectView = arr_objects.map(item => {
        if(typeof item[field1] === 'undefined' || typeof item[field2] === 'undefined'){
            return null
        }
        console.log(item)
        let obj = {"value": item[field1], "label": item[field2]}
        return obj
    })
    return dataEditToSelectView
}

