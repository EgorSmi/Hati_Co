import Select from 'react-select'

export default function SelectMain(props){

    const selectData = props.selectData
    const isDisabled = props.isDisabled ?? false
    const isLoading = props.isLoading ?? false
    const placeholder = props.placeholder

    return (
        <Select
          onChange={props.onChange}
          options={selectData}
          isDisabled={isDisabled}
          isLoading={isLoading}
          placeholder={placeholder}
          components={{
            IndicatorSeparator: () => null
          }}
        />
    );
}
